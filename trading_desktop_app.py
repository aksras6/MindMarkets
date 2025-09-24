from tkinter import ttk, filedialog, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import sys
import os
import json
import pandas as pd
from pathlib import Path
import subprocess
import webbrowser
from datetime import datetime, timedelta
import queue
import traceback
import configparser
import logging
from typing import Dict, List, Optional, Tuple
import time
import numpy as np
from itertools import product
import tkinter as tk

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import your existing modules
try:
    sys.path.append(".")
    from ibapi_appv1_patched import (
        build_strategy,
        STRATEGY_REGISTRY,
        STRATEGY_PARAM_KEYS,
        run_download_and_backtest,
        place_paper_orders_now,
        roll_daily_brackets_after_close,
        IbApp,
        latest_completed_daily_df,
        download_with_date_range,
        backtest_portfolio,
        save_csvs,
        _html_report,
        _json_report,
    )
    from strategies.base import BaseStrategy
    from dateutil.relativedelta import relativedelta
except ImportError as e:
    logger.error(f"Error importing trading modules: {e}")
    messagebox.showerror(
        "Import Error",
        "Could not import trading modules. Make sure your trading files are in the same directory.",
    )
    print("Warning: python-dateutil not installed. Walk-forward analysis may not work properly.")
    print("Install with: pip install python-dateutil")


class ConfigManager:
    """Manage application configuration with persistence"""

    def __init__(self):
        self.config_file = "trading_app_config.ini"
        self.config = configparser.ConfigParser()
        self.load_config()

    def load_config(self):
        if os.path.exists(self.config_file):
            self.config.read(self.config_file)
        else:
            self.set_defaults()

    def set_defaults(self):
        self.config["CONNECTION"] = {
            "host": "127.0.0.1",
            "port": "7497",
            "client_id": "44",
        }
        self.config["STRATEGY"] = {
            "current": "adx_squeeze",
            "len_channel": "20",
            "adx_len": "15",
            "adx_thresh": "20.0",
            "trade_pct": "15.0",
            "max_positions": "10",
            "max_exposure_pct": "100.0",
            "warmup_bars": "80",
        }
        self.config["DATA"] = {
            "start_date": "2020-01-01",
            "end_date": "",
            "timeframe": "1 day",
            "data_type": "TRADES",
        }
        self.config["GENERAL"] = {
            "output_dir": "output",
            "equity": "10000",
            "auto_save": "True",
            "theme": "clam",
        }
        self.save_config()

    def save_config(self):
        with open(self.config_file, "w") as f:
            self.config.write(f)

    def get(self, section, key, fallback=None):
        return self.config.get(section, key, fallback=fallback)

    def set(self, section, key, value):
        if section not in self.config:
            self.config.add_section(section)
        self.config[section][key] = str(value)
        if self.config.getboolean("GENERAL", "auto_save", fallback=True):
            self.save_config()


class OptimizationEngine:
    """Strategy parameter optimization engine"""

    def __init__(self, app):
        self.app = app
        self.results = []
        self.is_running = False
        self.stop_requested = False

    def generate_parameter_combinations(self, strategy_name: str, param_ranges: dict):
        """Generate all parameter combinations for optimization"""
        if strategy_name not in STRATEGY_PARAM_KEYS:
            return []

        valid_params = STRATEGY_PARAM_KEYS[strategy_name]
        combinations = []

        # Only use parameters that are valid for this strategy
        filtered_ranges = {k: v for k, v in param_ranges.items() if k in valid_params}

        if not filtered_ranges:
            return []

        # Generate all combinations
        keys = list(filtered_ranges.keys())
        values = list(filtered_ranges.values())

        for combination in product(*values):
            param_dict = dict(zip(keys, combination))
            combinations.append(param_dict)

        return combinations

    def run_optimization(
        self,
        symbols: List[str],
        strategy_name: str,
        param_ranges: dict,
        data_dict: dict,
        callback=None,
    ):
        """Run parameter optimization"""
        self.results = []
        self.is_running = True
        self.stop_requested = False

        combinations = self.generate_parameter_combinations(strategy_name, param_ranges)

        if not combinations:
            if callback:
                callback("No valid parameter combinations generated", "error")
            return

        total_combinations = len(combinations)

        for i, params in enumerate(combinations):
            if self.stop_requested:
                break

            try:
                # Build strategy with current parameters
                strategy = build_strategy(strategy_name, **params)

                # Run backtest
                trades_df, summary, equity_df = backtest_portfolio(data_dict, strategy)

                # Store results
                result = {
                    "params": params.copy(),
                    "summary": summary,
                    "trades_count": len(trades_df),
                    "total_return": summary.get("TotalReturnPct", 0),
                    "sharpe": summary.get("Sharpe", 0),
                    "max_dd": summary.get("MaxDDPct", 0),
                    "profit_factor": summary.get("ProfitFactor", 0),
                }

                self.results.append(result)

                if callback:
                    progress = (i + 1) / total_combinations * 100
                    callback(
                        f"Optimization progress: {progress:.1f}% ({i+1}/{total_combinations})",
                        "progress",
                    )

            except Exception as e:
                logger.error(f"Optimization error for params {params}: {e}")
                continue

        self.is_running = False

        if callback and not self.stop_requested:
            callback("Optimization completed successfully", "complete")

    def stop_optimization(self):
        """Stop running optimization"""
        self.stop_requested = True

    def get_best_results(self, metric="total_return", top_n=10):
        """Get best results sorted by specified metric"""
        if not self.results:
            return []

        # Sort by metric (handle 'Inf' values)
        def sort_key(result):
            value = result.get(metric, 0)
            if value == "Inf" or value == np.inf:
                return float("inf")
            if isinstance(value, str):
                try:
                    return float(value)
                except:
                    return 0
            return float(value) if value is not None else 0

        sorted_results = sorted(self.results, key=sort_key, reverse=True)
        return sorted_results[:top_n]

class WalkForwardEngine:
    """Walk-forward analysis engine"""

    def __init__(self, app):
        self.app = app
        self.results = []
        self.is_running = False
        self.stop_requested = False

    def run_walk_forward(
        self,
        symbols: List[str],
        strategy_name: str,
        param_ranges: dict,
        data_dict: dict,
        training_months: int = 12,
        testing_months: int = 3,
        reoptimize_months: int = 3,
        optimization_metric: str = "total_return",
        min_training_bars: int = 252,
        callback=None,
    ):
        """
        Run walk-forward analysis
        
        Args:
            symbols: List of symbols to test
            strategy_name: Strategy to use
            param_ranges: Parameter ranges for optimization
            data_dict: Historical data by symbol
            training_months: Months of training data
            testing_months: Months of testing data  
            reoptimize_months: How often to reoptimize (months)
            optimization_metric: Metric to optimize on
            min_training_bars: Minimum bars needed for training
            callback: Progress callback function
        """
        self.results = []
        self.is_running = True
        self.stop_requested = False

        try:
            # Get date range from data
            all_dates = []
            for df in data_dict.values():
                if not df.empty:
                    all_dates.extend(df.index.tolist())
            
            if not all_dates:
                if callback:
                    callback("No data available for walk-forward analysis", "error")
                return

            all_dates = sorted(set(all_dates))
            start_date = min(all_dates)
            end_date = max(all_dates)

            if callback:
                callback(f"Walk-forward period: {start_date.date()} to {end_date.date()}", "info")

            # Calculate walk-forward windows
            windows = self._calculate_wf_windows(
                start_date, end_date, training_months, testing_months, reoptimize_months
            )

            if not windows:
                if callback:
                    callback("No valid walk-forward windows found", "error")
                return

            total_windows = len(windows)
            if callback:
                callback(f"Generated {total_windows} walk-forward windows", "info")

            # Process each window
            for i, window in enumerate(windows):
                if self.stop_requested:
                    break

                train_start, train_end, test_start, test_end = window
                
                if callback:
                    progress = (i / total_windows) * 100
                    callback(
                        f"Window {i+1}/{total_windows}: Training {train_start.date()} to {train_end.date()}, "
                        f"Testing {test_start.date()} to {test_end.date()} ({progress:.1f}%)",
                        "progress"
                    )

                # Split data into training and testing
                train_data = self._filter_data_by_date(data_dict, train_start, train_end)
                test_data = self._filter_data_by_date(data_dict, test_start, test_end)

                # Check minimum data requirements
                train_bars = min(len(df) for df in train_data.values() if not df.empty) if train_data else 0
                if train_bars < min_training_bars:
                    if callback:
                        callback(f"Window {i+1}: Insufficient training data ({train_bars} bars)", "warning")
                    continue

                # Optimize on training data
                best_params = self._optimize_on_training_data(
                    train_data, strategy_name, param_ranges, optimization_metric
                )

                if not best_params:
                    if callback:
                        callback(f"Window {i+1}: Optimization failed", "warning")
                    continue

                # Test on out-of-sample data
                test_results = self._test_on_oos_data(
                    test_data, strategy_name, best_params
                )

                # Store results
                window_result = {
                    "window": i + 1,
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                    "best_params": best_params,
                    "oos_results": test_results,
                    "train_bars": train_bars,
                    "test_bars": min(len(df) for df in test_data.values() if not df.empty) if test_data else 0,
                }

                self.results.append(window_result)

            # Calculate aggregate statistics
            self._calculate_aggregate_stats()

            if callback:
                callback("Walk-forward analysis completed successfully", "complete")

        except Exception as e:
            if callback:
                callback(f"Walk-forward analysis failed: {str(e)}", "error")
            import traceback
            print(traceback.format_exc())

        finally:
            self.is_running = False

    def _calculate_wf_windows(self, start_date, end_date, training_months, testing_months, reoptimize_months):
        """Calculate walk-forward windows"""
        from dateutil.relativedelta import relativedelta
        
        windows = []
        current_start = start_date
        
        while current_start < end_date:
            # Training period
            train_start = current_start
            train_end = train_start + relativedelta(months=training_months)
            
            if train_end >= end_date:
                break
                
            # Testing period
            test_start = train_end
            test_end = test_start + relativedelta(months=testing_months)
            
            if test_end > end_date:
                test_end = end_date
                
            if test_start < test_end:  # Valid window
                windows.append((train_start, train_end, test_start, test_end))
            
            # Move forward by reoptimize_months
            current_start += relativedelta(months=reoptimize_months)
            
        return windows

    def _filter_data_by_date(self, data_dict, start_date, end_date):
        """Filter data dictionary by date range"""
        filtered = {}
        for symbol, df in data_dict.items():
            if df.empty:
                continue
            mask = (df.index >= start_date) & (df.index <= end_date)
            filtered_df = df[mask]
            if not filtered_df.empty:
                filtered[symbol] = filtered_df
        return filtered

    def _optimize_on_training_data(self, train_data, strategy_name, param_ranges, metric):
        """Run optimization on training data"""
        if not train_data:
            return None
            
        try:
            # Use the existing optimization engine
            combinations = self.app.optimizer.generate_parameter_combinations(
                strategy_name, param_ranges
            )
            
            if not combinations:
                return None

            best_result = None
            best_score = float('-inf')

            for params in combinations:
                try:
                    strategy = build_strategy(strategy_name, **params)
                    trades_df, summary, equity_df = backtest_portfolio(train_data, strategy)
                    
                    score = summary.get(self._metric_key_mapping().get(metric, metric), 0)
                    
                    # Handle special cases
                    if score == "Inf":
                        score = float('inf')
                    elif score == "NA":
                        score = 0
                    else:
                        score = float(score)

                    if score > best_score:
                        best_score = score
                        best_result = params.copy()

                except Exception as e:
                    print(f"Training optimization error for params {params}: {e}")
                    continue

            return best_result

        except Exception as e:
            print(f"Training optimization failed: {e}")
            return None

    def _test_on_oos_data(self, test_data, strategy_name, best_params):
        """Test optimized parameters on out-of-sample data"""
        if not test_data or not best_params:
            return None

        try:
            strategy = build_strategy(strategy_name, **best_params)
            trades_df, summary, equity_df = backtest_portfolio(test_data, strategy)
            return summary
        except Exception as e:
            print(f"Out-of-sample testing failed: {e}")
            return None

    def _calculate_aggregate_stats(self):
        """Calculate aggregate walk-forward statistics"""
        if not self.results:
            return

        # Collect all out-of-sample results
        oos_returns = []
        oos_trades = []
        window_count = 0

        for result in self.results:
            oos_results = result.get("oos_results")
            if oos_results:
                total_return = oos_results.get("TotalReturnPct", 0)
                trades = oos_results.get("Trades", 0)
                
                if isinstance(total_return, (int, float)):
                    oos_returns.append(total_return)
                if isinstance(trades, (int, float)):
                    oos_trades.append(trades)
                window_count += 1

        if oos_returns:
            # Calculate aggregate metrics
            self.aggregate_stats = {
                "total_windows": len(self.results),
                "successful_windows": window_count,
                "avg_oos_return": np.mean(oos_returns),
                "std_oos_return": np.std(oos_returns),
                "median_oos_return": np.median(oos_returns),
                "min_oos_return": min(oos_returns),
                "max_oos_return": max(oos_returns),
                "positive_windows": sum(1 for r in oos_returns if r > 0),
                "negative_windows": sum(1 for r in oos_returns if r < 0),
                "total_trades": sum(oos_trades),
                "avg_trades_per_window": np.mean(oos_trades) if oos_trades else 0,
            }
        else:
            self.aggregate_stats = {}

    def _metric_key_mapping(self):
        """Map optimization metric names to summary keys"""
        return {
            "total_return": "TotalReturnPct",
            "sharpe": "Sharpe", 
            "profit_factor": "ProfitFactor",
            "max_dd": "MaxDDPct",
        }

    def stop_walk_forward(self):
        """Stop running walk-forward analysis"""
        self.stop_requested = True

    def get_aggregate_results(self):
        """Get aggregate walk-forward results"""
        return getattr(self, 'aggregate_stats', {}), self.results

class TradingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("IBKR Trading Strategy Manager Pro")
        self.root.geometry("1600x1000")
        self.root.minsize(1400, 900)

        # Configuration manager
        self.config = ConfigManager()

        # Queue for thread communication
        self.message_queue = queue.Queue()

        # Data storage
        self.backtest_results = {}
        self.current_positions = []
        self.performance_history = []
        self.ib_app = None
        self.ib_connected = False
        self.data_cache = {}  # Cache for downloaded data

        # Optimization engine
        self.optimizer = OptimizationEngine(self)

        # Walk-forward engine
        self.walkforward = WalkForwardEngine(self)

        # Apply theme
        self.setup_theme()

        # Create main interface
        self.create_widgets()

        # Load saved configuration
        self.load_saved_config()

        # Start message processing and auto-refresh
        self.process_messages()
        self.auto_refresh_timer()

        # Set up keyboard shortcuts
        self.setup_shortcuts()

    def setup_theme(self):
        """Setup modern theme and styling"""
        style = ttk.Style()
        theme = self.config.get("GENERAL", "theme", "clam")

        try:
            style.theme_use(theme)
        except:
            style.theme_use("clam")

        # Custom styles
        style.configure("Title.TLabel", font=("Helvetica", 12, "bold"))
        style.configure("Success.TLabel", foreground="green")
        style.configure("Warning.TLabel", foreground="orange")
        style.configure("Error.TLabel", foreground="red")
        style.configure("Accent.TButton", font=("Helvetica", 10, "bold"))
        style.configure("Connected.TLabel", foreground="green")
        style.configure("Disconnected.TLabel", foreground="red")

    def setup_shortcuts(self):
        """Setup keyboard shortcuts"""
        self.root.bind("<Control-s>", lambda e: self.save_config_manually())
        self.root.bind("<Control-r>", lambda e: self.refresh_all())
        self.root.bind("<F5>", lambda e: self.refresh_all())
        self.root.bind("<Control-q>", lambda e: self.on_closing())
        self.root.bind("<Control-t>", lambda e: self.place_orders())
        self.root.bind("<Control-b>", lambda e: self.run_backtest())

    def create_widgets(self):
        # Create main menu
        self.create_menu()

        # Create toolbar
        self.create_toolbar()

        # Create main paned window
        self.main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel for configuration
        self.left_frame = ttk.Frame(self.main_paned, width=400)
        self.main_paned.add(self.left_frame, weight=1)

        # Right panel with tabs
        self.right_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(self.right_frame, weight=3)

        # Create left panel content
        self.create_config_panel()

        # Create notebook for tabs in right panel
        self.notebook = ttk.Notebook(self.right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Create tabs
        self.create_data_tab()
        self.create_backtest_tab()
        self.create_optimization_tab()
        self.create_walkforward_tab()
        self.create_trading_tab()
        self.create_portfolio_tab()
        self.create_analytics_tab()
        self.create_logs_tab()

        # Status bar with multiple sections
        self.create_status_bar()

    def create_menu(self):
        """Create application menu"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Config...", command=self.load_config_file)
        file_menu.add_command(label="Save Config...", command=self.save_config_file)
        file_menu.add_separator()
        file_menu.add_command(label="Export Results...", command=self.export_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)

        # Connection menu
        conn_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Connection", menu=conn_menu)
        conn_menu.add_command(label="Connect to IB", command=self.connect_ib)
        conn_menu.add_command(label="Disconnect from IB", command=self.disconnect_ib)
        conn_menu.add_command(label="Test Connection", command=self.test_connection)

        # Data menu
        data_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Data", menu=data_menu)
        data_menu.add_command(label="Download Data", command=self.download_data)
        data_menu.add_command(label="Clear Cache", command=self.clear_cache)
        data_menu.add_command(label="Validate Symbols", command=self.validate_symbols)

        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Run Optimization", command=self.run_optimization)
        tools_menu.add_command(label="Run Walk-Forward", command=self.start_walk_forward)
        tools_menu.add_command(
            label="Strategy Comparison", command=self.compare_strategies
        )
        tools_menu.add_separator()
        tools_menu.add_command(label="Debug Strategy Issues", command=self.debug_strategy_issues)
     
            


        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="User Guide", command=self.show_help)
        help_menu.add_command(label="Keyboard Shortcuts", command=self.show_shortcuts)
        help_menu.add_command(label="About", command=self.show_about)

    def create_toolbar(self):
        """Create toolbar with quick actions"""
        toolbar = ttk.Frame(self.root)
        toolbar.pack(fill=tk.X, padx=5, pady=2)

        # Connection buttons
        ttk.Button(toolbar, text="🔌 Connect", command=self.connect_ib).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(toolbar, text="🔌 Disconnect", command=self.disconnect_ib).pack(
            side=tk.LEFT, padx=2
        )

        ttk.Separator(toolbar, orient="vertical").pack(side=tk.LEFT, padx=5, fill=tk.Y)

        # Data and analysis buttons
        ttk.Button(toolbar, text="📥 Download Data", command=self.download_data).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(toolbar, text="🔄 Refresh", command=self.refresh_all).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(toolbar, text="📊 Quick Backtest", command=self.quick_backtest).pack(
            side=tk.LEFT, padx=2
        )

        ttk.Separator(toolbar, orient="vertical").pack(side=tk.LEFT, padx=5, fill=tk.Y)

        # Trading buttons
        ttk.Button(toolbar, text="📈 Place Orders", command=self.place_orders).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(toolbar, text="🎯 Optimize", command=self.run_optimization).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(toolbar, text="📊 Walk-Forward", command=self.start_walk_forward).pack(  # Add this
            side=tk.LEFT, padx=2
        )

        # Connection status indicator
        self.connection_status = ttk.Label(
            toolbar, text="● Disconnected", style="Disconnected.TLabel"
        )
        self.connection_status.pack(side=tk.RIGHT, padx=5)

    def create_config_panel(self):
        """Create configuration panel in left frame"""
        # Title
        title_label = ttk.Label(
            self.left_frame, text="Configuration", style="Title.TLabel"
        )
        title_label.pack(pady=5)

        # Connection Settings
        conn_frame = ttk.LabelFrame(self.left_frame, text="IB Connection")
        conn_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(conn_frame, text="Host:").grid(
            row=0, column=0, sticky=tk.W, padx=5, pady=2
        )
        self.host_var = tk.StringVar()
        ttk.Entry(conn_frame, textvariable=self.host_var, width=20).grid(
            row=0, column=1, padx=5, pady=2
        )

        ttk.Label(conn_frame, text="Port:").grid(
            row=1, column=0, sticky=tk.W, padx=5, pady=2
        )
        self.port_var = tk.StringVar()
        ttk.Entry(conn_frame, textvariable=self.port_var, width=20).grid(
            row=1, column=1, padx=5, pady=2
        )

        ttk.Label(conn_frame, text="Client ID:").grid(
            row=2, column=0, sticky=tk.W, padx=5, pady=2
        )
        self.client_id_var = tk.StringVar()
        ttk.Entry(conn_frame, textvariable=self.client_id_var, width=20).grid(
            row=2, column=1, padx=5, pady=2
        )

        # Connection buttons
        conn_buttons = ttk.Frame(conn_frame)
        conn_buttons.grid(row=3, column=0, columnspan=2, pady=5)
        ttk.Button(conn_buttons, text="Connect", command=self.connect_ib).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(conn_buttons, text="Disconnect", command=self.disconnect_ib).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(conn_buttons, text="Test", command=self.test_connection).pack(
            side=tk.LEFT, padx=2
        )

        # Data Settings
        data_frame = ttk.LabelFrame(self.left_frame, text="Data Settings")
        data_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(data_frame, text="Start Date:").grid(
            row=0, column=0, sticky=tk.W, padx=5, pady=2
        )
        self.start_date_var = tk.StringVar()
        ttk.Entry(data_frame, textvariable=self.start_date_var, width=20).grid(
            row=0, column=1, padx=5, pady=2
        )

        ttk.Label(data_frame, text="End Date:").grid(
            row=1, column=0, sticky=tk.W, padx=5, pady=2
        )
        self.end_date_var = tk.StringVar()
        ttk.Entry(data_frame, textvariable=self.end_date_var, width=20).grid(
            row=1, column=1, padx=5, pady=2
        )

        ttk.Label(data_frame, text="Timeframe:").grid(
            row=2, column=0, sticky=tk.W, padx=5, pady=2
        )
        self.timeframe_var = tk.StringVar()
        timeframe_combo = ttk.Combobox(
            data_frame,
            textvariable=self.timeframe_var,
            width=18,
            values=[
                "1 min",
                "5 mins",
                "15 mins",
                "30 mins",
                "1 hour",
                "1 day",
                "1 week",
            ],
        )
        timeframe_combo.grid(row=2, column=1, padx=5, pady=2)

        ttk.Label(data_frame, text="Data Type:").grid(
            row=3, column=0, sticky=tk.W, padx=5, pady=2
        )
        self.data_type_var = tk.StringVar()
        data_type_combo = ttk.Combobox(
            data_frame,
            textvariable=self.data_type_var,
            width=18,
            values=["TRADES", "MIDPOINT", "BID", "ASK", "ADJUSTED_LAST"],
        )
        data_type_combo.grid(row=3, column=1, padx=5, pady=2)

        # Strategy Settings with save button
        strat_frame = ttk.LabelFrame(self.left_frame, text="Strategy")
        strat_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(strat_frame, text="Strategy:").grid(
            row=0, column=0, sticky=tk.W, padx=5, pady=2
        )
        self.strategy_var = tk.StringVar()
        strategy_combo = ttk.Combobox(
            strat_frame,
            textvariable=self.strategy_var,
            values=list(STRATEGY_REGISTRY.keys()),
            state="readonly",
            width=18,
        )
        strategy_combo.grid(row=0, column=1, padx=5, pady=2)
        strategy_combo.bind("<<ComboboxSelected>>", self.on_strategy_change)

        # Strategy parameters in a scrollable frame
        params_container = ttk.Frame(strat_frame)
        params_container.grid(row=1, column=0, columnspan=2, sticky="ew", pady=5)

        self.params_canvas = tk.Canvas(params_container, height=200)
        self.params_scrollbar = ttk.Scrollbar(
            params_container, orient="vertical", command=self.params_canvas.yview
        )
        self.params_frame = ttk.Frame(self.params_canvas)

        self.params_canvas.configure(yscrollcommand=self.params_scrollbar.set)
        self.params_canvas.pack(side="left", fill="both", expand=True)
        self.params_scrollbar.pack(side="right", fill="y")

        self.params_canvas.create_window((0, 0), window=self.params_frame, anchor="nw")
        self.params_frame.bind(
            "<Configure>",
            lambda e: self.params_canvas.configure(
                scrollregion=self.params_canvas.bbox("all")
            ),
        )

        self.param_vars = {}

        # Save configuration button
        ttk.Button(
            strat_frame, text="Save Config", command=self.save_config_manually
        ).grid(row=2, column=0, columnspan=2, pady=5)

        # Symbols with enhanced features
        symbols_frame = ttk.LabelFrame(self.left_frame, text="Symbols")
        symbols_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        symbols_toolbar = ttk.Frame(symbols_frame)
        symbols_toolbar.pack(fill=tk.X, padx=5, pady=2)

        ttk.Button(
            symbols_toolbar, text="📁 Load", command=self.load_symbols_file
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(
            symbols_toolbar, text="💾 Save", command=self.save_symbols_file
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(
            symbols_toolbar, text="🔄 Default", command=self.use_default_symbols
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(
            symbols_toolbar, text="✓ Validate", command=self.validate_symbols
        ).pack(side=tk.LEFT, padx=2)

        self.symbols_text = scrolledtext.ScrolledText(symbols_frame, height=8, width=30)
        self.symbols_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Symbol count label
        self.symbol_count_label = ttk.Label(symbols_frame, text="Symbols: 0")
        self.symbol_count_label.pack(pady=2)

        self.symbols_text.bind("<KeyRelease>", self.update_symbol_count)


    def on_wf_strategy_change(self, event=None):
        """Handle walk-forward strategy change"""
        self.create_wf_optimization_params()

    def create_wf_optimization_params(self):
        """Create walk-forward parameter range inputs"""
        # Clear existing widgets
        for widget in self.wf_param_ranges_frame.winfo_children():
            widget.destroy()
        self.wf_param_range_vars.clear()

        strategy = self.wf_strategy_var.get()
        if strategy not in STRATEGY_PARAM_KEYS:
            return

        params = STRATEGY_PARAM_KEYS[strategy]

        # Default ranges for common parameters  
        default_ranges = {
            "len_channel": "15,20,25",
            "adx_len": "10,14,20", 
            "adx_thresh": "15,20,25,30",
            "rsi_len": "10,14,20",
            "rsi_thresh": "50,55,60,65",
            "trade_pct": "10,15,20",
            "max_positions": "5,8,10,15",
        }

        row = 0
        for param in sorted(params):
            ttk.Label(
                self.wf_param_ranges_frame, text=f"{param.replace('_', ' ').title()}:"
            ).grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)

            var = tk.StringVar(value=default_ranges.get(param, ""))
            self.wf_param_range_vars[param] = var
            entry = ttk.Entry(self.wf_param_ranges_frame, textvariable=var, width=30)
            entry.grid(row=row, column=1, padx=5, pady=2)

            ttk.Label(self.wf_param_ranges_frame, text="(comma-separated values)").grid(
                row=row, column=2, sticky=tk.W, padx=5, pady=2
            )

            row += 1

    def start_walk_forward(self):
        """Start walk-forward analysis"""
        if not self.data_cache:
            messagebox.showwarning("Warning", "No data available. Download data first.")
            return

        strategy = self.wf_strategy_var.get()
        if not strategy:
            messagebox.showwarning("Warning", "Please select a strategy")
            return

        # Parse parameter ranges
        param_ranges = {}
        for param, var in self.wf_param_range_vars.items():
            range_str = var.get().strip()
            if range_str:
                try:
                    values = [
                        float(x.strip()) if "." in x.strip() else int(x.strip())
                        for x in range_str.split(",")
                    ]
                    param_ranges[param] = values
                except ValueError:
                    messagebox.showerror(
                        "Error", f"Invalid range for {param}: {range_str}"
                    )
                    return

        if not param_ranges:
            messagebox.showwarning("Warning", "Please specify parameter ranges")
            return

        # Get time period settings
        try:
            training_months = int(self.wf_training_months_var.get())
            testing_months = int(self.wf_testing_months_var.get())
            reoptimize_months = int(self.wf_reoptimize_months_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid time period values")
            return

        self.update_status("Starting walk-forward analysis...")
        self.wf_progress.config(mode="determinate", maximum=100, value=0)

        def wf_callback(message, msg_type):
            if msg_type == "progress":
                if "%" in message:
                    try:
                        progress = float(message.split("(")[1].split("%")[0])
                        self.wf_progress.config(value=progress)
                    except:
                        pass
            self.message_queue.put((f"wf_{msg_type}", message))

        def run_wf():
            self.walkforward.run_walk_forward(
                symbols=list(self.data_cache.keys()),
                strategy_name=strategy,
                param_ranges=param_ranges,
                data_dict=self.data_cache,
                training_months=training_months,
                testing_months=testing_months, 
                reoptimize_months=reoptimize_months,
                optimization_metric=self.wf_metric_var.get(),
                callback=wf_callback,
            )

            # Display results
            aggregate_stats, window_results = self.walkforward.get_aggregate_results()
            self.message_queue.put(("wf_results", (aggregate_stats, window_results)))

        self.run_in_thread(run_wf)

    def stop_walk_forward(self):
        """Stop walk-forward analysis"""
        self.walkforward.stop_walk_forward()
        self.update_status("Walk-forward analysis stopped")

    def display_wf_results(self, aggregate_stats, window_results):
        """Display walk-forward results"""
        # Clear previous results
        self.wf_results_tree.delete(*self.wf_results_tree.get_children())
        self.wf_summary_tree.delete(*self.wf_summary_tree.get_children())

        # Display window results
        for result in window_results:
            window_num = result.get("window", "")
            train_period = f"{result['train_start'].strftime('%Y-%m-%d')} to {result['train_end'].strftime('%Y-%m-%d')}"
            test_period = f"{result['test_start'].strftime('%Y-%m-%d')} to {result['test_end'].strftime('%Y-%m-%d')}"
            
            best_params = result.get("best_params", {})
            params_str = ", ".join([f"{k}={v}" for k, v in best_params.items()])
            
            oos_results = result.get("oos_results", {})
            oos_return = oos_results.get("TotalReturnPct", "N/A")
            oos_sharpe = oos_results.get("Sharpe", "N/A") 
            oos_trades = oos_results.get("Trades", "N/A")
            
            train_bars = result.get("train_bars", "N/A")
            test_bars = result.get("test_bars", "N/A")

            self.wf_results_tree.insert(
                "",
                "end",
                text=str(window_num),
                values=(
                    train_period,
                    test_period,
                    params_str,
                    f"{oos_return:.2f}" if isinstance(oos_return, (int, float)) else str(oos_return),
                    f"{oos_sharpe:.3f}" if isinstance(oos_sharpe, (int, float)) else str(oos_sharpe),
                    str(oos_trades),
                    str(train_bars),
                    str(test_bars),
                ),
            )

        # Display aggregate statistics
        for key, value in aggregate_stats.items():
            self.wf_summary_tree.insert(
                "", "end", text=key.replace('_', ' ').title(), values=(str(value),)
            )

    def export_wf_results(self):
        """Export walk-forward results"""
        if not hasattr(self.walkforward, 'results') or not self.walkforward.results:
            messagebox.showwarning("Warning", "No walk-forward results to export")
            return

        filename = filedialog.asksaveasfilename(
            title="Export Walk-Forward Results",
            defaultextension=".xlsx",
            filetypes=[
                ("Excel files", "*.xlsx"),
                ("CSV files", "*.csv"),
                ("All files", "*.*"),
            ],
        )

        if filename:
            try:
                # Convert results to DataFrame
                data = []
                for result in self.walkforward.results:
                    row = {
                        "window": result.get("window"),
                        "train_start": result.get("train_start"),
                        "train_end": result.get("train_end"),  
                        "test_start": result.get("test_start"),
                        "test_end": result.get("test_end"),
                        "train_bars": result.get("train_bars"),
                        "test_bars": result.get("test_bars"),
                    }
                    
                    # Add best parameters
                    best_params = result.get("best_params", {})
                    for param, value in best_params.items():
                        row[f"param_{param}"] = value
                    
                    # Add OOS results
                    oos_results = result.get("oos_results", {})
                    for metric, value in oos_results.items():
                        row[f"oos_{metric}"] = value
                        
                    data.append(row)

                df = pd.DataFrame(data)

                if filename.endswith(".xlsx"):
                    with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
                        df.to_excel(writer, sheet_name="Walk_Forward_Results", index=False)
                        
                        # Add aggregate stats sheet
                        aggregate_stats, _ = self.walkforward.get_aggregate_results()
                        if aggregate_stats:
                            agg_df = pd.DataFrame([aggregate_stats])
                            agg_df.to_excel(writer, sheet_name="Aggregate_Stats", index=False)
                else:
                    df.to_csv(filename, index=False)

                messagebox.showinfo("Success", f"Walk-forward results exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {e}")

    def create_data_tab(self):
        """Data management tab"""
        self.data_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.data_frame, text="📥 Data")

        # Data controls
        controls_frame = ttk.LabelFrame(self.data_frame, text="Data Management")
        controls_frame.pack(fill=tk.X, padx=5, pady=5)

        # Download controls
        download_frame = ttk.Frame(controls_frame)
        download_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(
            download_frame,
            text="📥 Download All Data",
            command=self.download_data,
            style="Accent.TButton",
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            download_frame, text="🔄 Update Data", command=self.update_data
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(download_frame, text="🗑️ Clear Cache", command=self.clear_cache).pack(
            side=tk.LEFT, padx=5
        )

        # Progress bar
        self.data_progress = ttk.Progressbar(controls_frame, mode="determinate")
        self.data_progress.pack(fill=tk.X, padx=5, pady=5)

        self.data_progress_label = ttk.Label(controls_frame, text="Ready")
        self.data_progress_label.pack(pady=2)

        # Data status table
        status_frame = ttk.LabelFrame(self.data_frame, text="Data Status")
        status_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.data_status_tree = ttk.Treeview(
            status_frame,
            columns=("Status", "Last_Updated", "Bars", "Date_Range"),
            show="tree headings",
        )
        self.data_status_tree.heading("#0", text="Symbol")
        self.data_status_tree.heading("Status", text="Status")
        self.data_status_tree.heading("Last_Updated", text="Last Updated")
        self.data_status_tree.heading("Bars", text="Bars")
        self.data_status_tree.heading("Date_Range", text="Date Range")

        # Add scrollbar to data status
        data_scrollbar = ttk.Scrollbar(
            status_frame, orient="vertical", command=self.data_status_tree.yview
        )
        self.data_status_tree.configure(yscrollcommand=data_scrollbar.set)

        self.data_status_tree.pack(
            side="left", fill="both", expand=True, padx=5, pady=5
        )
        data_scrollbar.pack(side="right", fill="y")

        # Data output
        self.data_output = scrolledtext.ScrolledText(self.data_frame, height=8)
        self.data_output.pack(fill=tk.X, padx=5, pady=5)

    def create_optimization_tab(self):
        """Strategy optimization tab"""
        self.optimization_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.optimization_frame, text="🎯 Optimization")

        # Create paned window for optimization
        opt_paned = ttk.PanedWindow(self.optimization_frame, orient=tk.VERTICAL)
        opt_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Top frame for controls
        opt_controls_frame = ttk.Frame(opt_paned)
        opt_paned.add(opt_controls_frame, weight=1)

        # Optimization controls
        controls_frame = ttk.LabelFrame(
            opt_controls_frame, text="Optimization Settings"
        )
        controls_frame.pack(fill=tk.X, padx=5, pady=5)

        # Strategy selection for optimization
        ttk.Label(controls_frame, text="Strategy:").grid(
            row=0, column=0, sticky=tk.W, padx=5, pady=2
        )
        self.opt_strategy_var = tk.StringVar()
        opt_strategy_combo = ttk.Combobox(
            controls_frame,
            textvariable=self.opt_strategy_var,
            values=list(STRATEGY_REGISTRY.keys()),
            state="readonly",
        )
        opt_strategy_combo.grid(row=0, column=1, padx=5, pady=2)
        opt_strategy_combo.bind("<<ComboboxSelected>>", self.on_opt_strategy_change)

        # Optimization metric
        ttk.Label(controls_frame, text="Optimize for:").grid(
            row=0, column=2, sticky=tk.W, padx=5, pady=2
        )
        self.opt_metric_var = tk.StringVar(value="total_return")
        opt_metric_combo = ttk.Combobox(
            controls_frame,
            textvariable=self.opt_metric_var,
            values=["total_return", "sharpe", "profit_factor", "max_dd"],
            state="readonly",
        )
        opt_metric_combo.grid(row=0, column=3, padx=5, pady=2)

        # Parameter ranges frame
        self.param_ranges_frame = ttk.LabelFrame(
            opt_controls_frame, text="Parameter Ranges"
        )
        self.param_ranges_frame.pack(fill=tk.X, padx=5, pady=5)

        self.param_range_vars = {}
        self.create_optimization_params()

        # Control buttons
        button_frame = ttk.Frame(opt_controls_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(
            button_frame,
            text="🚀 Start Optimization",
            command=self.start_optimization,
            style="Accent.TButton",
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            button_frame, text="⏹️ Stop Optimization", command=self.stop_optimization
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            button_frame,
            text="📊 Export Results",
            command=self.export_optimization_results,
        ).pack(side=tk.LEFT, padx=5)

        # Progress
        self.opt_progress = ttk.Progressbar(opt_controls_frame, mode="determinate")
        self.opt_progress.pack(fill=tk.X, padx=5, pady=5)

        self.opt_progress_label = ttk.Label(opt_controls_frame, text="Ready")
        self.opt_progress_label.pack(pady=2)

        # Bottom frame for results
        opt_results_frame = ttk.Frame(opt_paned)
        opt_paned.add(opt_results_frame, weight=2)

        # Results table
        results_frame = ttk.LabelFrame(opt_results_frame, text="Optimization Results")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.opt_results_tree = ttk.Treeview(
            results_frame,
            columns=(
                "Total_Return",
                "Sharpe",
                "Max_DD",
                "Profit_Factor",
                "Trades",
                "Parameters",
            ),
            show="tree headings",
        )
        self.opt_results_tree.heading("#0", text="Rank")
        self.opt_results_tree.heading("Total_Return", text="Total Return %")
        self.opt_results_tree.heading("Sharpe", text="Sharpe Ratio")
        self.opt_results_tree.heading("Max_DD", text="Max DD %")
        self.opt_results_tree.heading("Profit_Factor", text="Profit Factor")
        self.opt_results_tree.heading("Trades", text="Trades")
        self.opt_results_tree.heading("Parameters", text="Parameters")

        # Column widths
        self.opt_results_tree.column("#0", width=50)
        self.opt_results_tree.column("Total_Return", width=100)
        self.opt_results_tree.column("Sharpe", width=80)
        self.opt_results_tree.column("Max_DD", width=80)
        self.opt_results_tree.column("Profit_Factor", width=100)
        self.opt_results_tree.column("Trades", width=60)
        self.opt_results_tree.column("Parameters", width=300)

        # Add scrollbar to optimization results
        opt_scrollbar = ttk.Scrollbar(
            results_frame, orient="vertical", command=self.opt_results_tree.yview
        )
        self.opt_results_tree.configure(yscrollcommand=opt_scrollbar.set)

        self.opt_results_tree.pack(
            side="left", fill="both", expand=True, padx=5, pady=5
        )
        opt_scrollbar.pack(side="right", fill="y")

    def create_walkforward_tab(self):
        """Walk-forward analysis tab"""
        self.walkforward_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.walkforward_frame, text="📈 Walk-Forward")

        # Create paned window for walk-forward
        wf_paned = ttk.PanedWindow(self.walkforward_frame, orient=tk.VERTICAL)
        wf_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Top frame for controls
        wf_controls_frame = ttk.Frame(wf_paned)
        wf_paned.add(wf_controls_frame, weight=1)

        # Walk-forward controls
        controls_frame = ttk.LabelFrame(wf_controls_frame, text="Walk-Forward Settings")
        controls_frame.pack(fill=tk.X, padx=5, pady=5)

        # Strategy and metric selection
        row1 = ttk.Frame(controls_frame)
        row1.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(row1, text="Strategy:").pack(side=tk.LEFT, padx=5)
        self.wf_strategy_var = tk.StringVar()
        wf_strategy_combo = ttk.Combobox(
            row1,
            textvariable=self.wf_strategy_var,
            values=list(STRATEGY_REGISTRY.keys()),
            state="readonly",
            width=15
        )
        wf_strategy_combo.pack(side=tk.LEFT, padx=5)
        wf_strategy_combo.bind("<<ComboboxSelected>>", self.on_wf_strategy_change)

        ttk.Label(row1, text="Optimize for:").pack(side=tk.LEFT, padx=15)
        self.wf_metric_var = tk.StringVar(value="total_return")
        wf_metric_combo = ttk.Combobox(
            row1,
            textvariable=self.wf_metric_var,
            values=["total_return", "sharpe", "profit_factor", "max_dd"],
            state="readonly",
            width=15
        )
        wf_metric_combo.pack(side=tk.LEFT, padx=5)

        # Time period settings
        row2 = ttk.Frame(controls_frame)
        row2.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(row2, text="Training (months):").pack(side=tk.LEFT, padx=5)
        self.wf_training_months_var = tk.StringVar(value="12")
        ttk.Entry(row2, textvariable=self.wf_training_months_var, width=8).pack(side=tk.LEFT, padx=5)

        ttk.Label(row2, text="Testing (months):").pack(side=tk.LEFT, padx=15)
        self.wf_testing_months_var = tk.StringVar(value="3")
        ttk.Entry(row2, textvariable=self.wf_testing_months_var, width=8).pack(side=tk.LEFT, padx=5)

        ttk.Label(row2, text="Reoptimize every (months):").pack(side=tk.LEFT, padx=15)
        self.wf_reoptimize_months_var = tk.StringVar(value="3")
        ttk.Entry(row2, textvariable=self.wf_reoptimize_months_var, width=8).pack(side=tk.LEFT, padx=5)

        # Parameter ranges frame (reuse from optimization)
        self.wf_param_ranges_frame = ttk.LabelFrame(wf_controls_frame, text="Parameter Ranges")
        self.wf_param_ranges_frame.pack(fill=tk.X, padx=5, pady=5)

        self.wf_param_range_vars = {}
        
        # Control buttons
        button_frame = ttk.Frame(wf_controls_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(
            button_frame,
            text="🚀 Start Walk-Forward",
            command=self.start_walk_forward,
            style="Accent.TButton",
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            button_frame, text="⏹️ Stop Analysis", command=self.stop_walk_forward
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            button_frame, text="📊 Export Results", command=self.export_wf_results
        ).pack(side=tk.LEFT, padx=5)

        # Progress
        self.wf_progress = ttk.Progressbar(wf_controls_frame, mode="determinate")
        self.wf_progress.pack(fill=tk.X, padx=5, pady=5)

        self.wf_progress_label = ttk.Label(wf_controls_frame, text="Ready")
        self.wf_progress_label.pack(pady=2)

        # Bottom frame for results
        wf_results_frame = ttk.Frame(wf_paned)
        wf_paned.add(wf_results_frame, weight=2)

        # Results notebook
        wf_results_notebook = ttk.Notebook(wf_results_frame)
        wf_results_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Window results tab
        window_results_frame = ttk.Frame(wf_results_notebook)
        wf_results_notebook.add(window_results_frame, text="Window Results")

        self.wf_results_tree = ttk.Treeview(
            window_results_frame,
            columns=(
                "Train_Period", "Test_Period", "Best_Params", "OOS_Return", 
                "OOS_Sharpe", "OOS_Trades", "Train_Bars", "Test_Bars"
            ),
            show="tree headings",
        )
        self.wf_results_tree.heading("#0", text="Window")
        self.wf_results_tree.heading("Train_Period", text="Training Period")
        self.wf_results_tree.heading("Test_Period", text="Testing Period") 
        self.wf_results_tree.heading("Best_Params", text="Best Parameters")
        self.wf_results_tree.heading("OOS_Return", text="OOS Return %")
        self.wf_results_tree.heading("OOS_Sharpe", text="OOS Sharpe")
        self.wf_results_tree.heading("OOS_Trades", text="OOS Trades")
        self.wf_results_tree.heading("Train_Bars", text="Train Bars")
        self.wf_results_tree.heading("Test_Bars", text="Test Bars")

        # Column widths
        self.wf_results_tree.column("#0", width=60)
        self.wf_results_tree.column("Train_Period", width=120)
        self.wf_results_tree.column("Test_Period", width=120)
        self.wf_results_tree.column("Best_Params", width=200)
        self.wf_results_tree.column("OOS_Return", width=100)
        self.wf_results_tree.column("OOS_Sharpe", width=80)
        self.wf_results_tree.column("OOS_Trades", width=80)
        self.wf_results_tree.column("Train_Bars", width=80)
        self.wf_results_tree.column("Test_Bars", width=80)

        wf_scrollbar = ttk.Scrollbar(
            window_results_frame, orient="vertical", command=self.wf_results_tree.yview
        )
        self.wf_results_tree.configure(yscrollcommand=wf_scrollbar.set)

        self.wf_results_tree.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        wf_scrollbar.pack(side="right", fill="y")

        # Summary tab
        summary_frame = ttk.Frame(wf_results_notebook)
        wf_results_notebook.add(summary_frame, text="Summary")

        self.wf_summary_tree = ttk.Treeview(
            summary_frame, columns=("Value",), show="tree headings"
        )
        self.wf_summary_tree.heading("#0", text="Metric")
        self.wf_summary_tree.heading("Value", text="Value")
        self.wf_summary_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_backtest_tab(self):
        """Enhanced backtest tab with real-time results"""
        self.backtest_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.backtest_frame, text="📊 Backtest")

        # Create paned window for backtest
        backtest_paned = ttk.PanedWindow(self.backtest_frame, orient=tk.VERTICAL)
        backtest_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Top frame for controls and progress
        top_frame = ttk.Frame(backtest_paned)
        backtest_paned.add(top_frame, weight=1)

        # Controls
        controls_frame = ttk.LabelFrame(top_frame, text="Backtest Controls")
        controls_frame.pack(fill=tk.X, padx=5, pady=5)

        controls_grid = ttk.Frame(controls_frame)
        controls_grid.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(controls_grid, text="Output Dir:").grid(
            row=0, column=0, sticky=tk.W, padx=5, pady=2
        )
        self.output_dir_var = tk.StringVar()
        ttk.Entry(controls_grid, textvariable=self.output_dir_var, width=30).grid(
            row=0, column=1, padx=5, pady=2
        )
        ttk.Button(controls_grid, text="📁", command=self.browse_output_dir).grid(
            row=0, column=2, padx=2, pady=2
        )

        options_frame = ttk.Frame(controls_grid)
        options_frame.grid(row=1, column=0, columnspan=3, sticky="w", pady=5)

        self.html_report_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            options_frame, text="HTML Report", variable=self.html_report_var
        ).pack(side=tk.LEFT, padx=10)

        self.json_report_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            options_frame, text="JSON Report", variable=self.json_report_var
        ).pack(side=tk.LEFT, padx=10)

        self.auto_open_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            options_frame, text="Auto-open Results", variable=self.auto_open_var
        ).pack(side=tk.LEFT, padx=10)

        # Action buttons
        button_frame = ttk.Frame(controls_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(
            button_frame,
            text="🚀 Run Full Backtest",
            command=self.run_backtest,
            style="Accent.TButton",
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            button_frame, text="⚡ Quick Test", command=self.quick_backtest
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            button_frame, text="📈 View Results", command=self.view_html_report
        ).pack(side=tk.LEFT, padx=5)

        # Progress with details
        progress_frame = ttk.LabelFrame(top_frame, text="Progress")
        progress_frame.pack(fill=tk.X, padx=5, pady=5)

        self.backtest_progress = ttk.Progressbar(progress_frame, mode="indeterminate")
        self.backtest_progress.pack(fill=tk.X, padx=5, pady=2)

        self.progress_label = ttk.Label(progress_frame, text="Ready")
        self.progress_label.pack(pady=2)

        # Bottom frame for output and results
        bottom_frame = ttk.Frame(backtest_paned)
        backtest_paned.add(bottom_frame, weight=2)

        results_notebook = ttk.Notebook(bottom_frame)
        results_notebook.pack(fill=tk.BOTH, expand=True)

        # Output tab
        output_frame = ttk.Frame(results_notebook)
        results_notebook.add(output_frame, text="Output")
        self.backtest_output = scrolledtext.ScrolledText(output_frame, height=15)
        self.backtest_output.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Quick results tab
        quick_results_frame = ttk.Frame(results_notebook)
        results_notebook.add(quick_results_frame, text="Quick Results")
        self.quick_results_tree = ttk.Treeview(
            quick_results_frame, columns=("Value",), show="tree headings"
        )
        self.quick_results_tree.heading("#0", text="Metric")
        self.quick_results_tree.heading("Value", text="Value")
        self.quick_results_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_trading_tab(self):
        """Enhanced trading tab with position monitoring"""
        self.trading_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.trading_frame, text="🎯 Trading")

        # Create paned window
        trading_paned = ttk.PanedWindow(self.trading_frame, orient=tk.VERTICAL)
        trading_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Top frame for controls
        controls_container = ttk.Frame(trading_paned)
        trading_paned.add(controls_container, weight=1)

        # Trading Controls
        controls_frame = ttk.LabelFrame(
            controls_container, text="Paper Trading Controls"
        )
        controls_frame.pack(fill=tk.X, padx=5, pady=5)

        # Account settings
        account_frame = ttk.Frame(controls_frame)
        account_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(account_frame, text="Account Equity:").pack(side=tk.LEFT, padx=5)
        self.equity_var = tk.StringVar()
        ttk.Entry(account_frame, textvariable=self.equity_var, width=15).pack(
            side=tk.LEFT, padx=5
        )

        # Action buttons with enhanced features
        button_frame = ttk.Frame(controls_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(
            button_frame,
            text="🚀 Place Orders",
            command=self.place_orders,
            style="Accent.TButton",
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="🔄 Roll Orders", command=self.roll_orders).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(button_frame, text="👁️ Preview", command=self.preview_orders).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(
            button_frame, text="📊 Current Signals", command=self.show_current_signals
        ).pack(side=tk.LEFT, padx=5)

        # Bottom frame for output and positions
        output_container = ttk.Frame(trading_paned)
        trading_paned.add(output_container, weight=2)

        trading_notebook = ttk.Notebook(output_container)
        trading_notebook.pack(fill=tk.BOTH, expand=True)

        # Output tab
        output_frame = ttk.Frame(trading_notebook)
        trading_notebook.add(output_frame, text="Output")
        self.trading_output = scrolledtext.ScrolledText(output_frame, height=15)
        self.trading_output.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Current signals tab
        signals_frame = ttk.Frame(trading_notebook)
        trading_notebook.add(signals_frame, text="Current Signals")
        self.signals_tree = ttk.Treeview(
            signals_frame,
            columns=("Entry", "Stop", "Qty", "Risk", "Eligible"),
            show="tree headings",
        )
        self.signals_tree.heading("#0", text="Symbol")
        self.signals_tree.heading("Entry", text="Entry Price")
        self.signals_tree.heading("Stop", text="Stop Loss")
        self.signals_tree.heading("Qty", text="Quantity")
        self.signals_tree.heading("Risk", text="Risk")
        self.signals_tree.heading("Eligible", text="Eligible")
        self.signals_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_portfolio_tab(self):
        """Portfolio monitoring and position management"""
        self.portfolio_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.portfolio_frame, text="💼 Portfolio")

        # Portfolio summary
        summary_frame = ttk.LabelFrame(self.portfolio_frame, text="Portfolio Summary")
        summary_frame.pack(fill=tk.X, padx=5, pady=5)

        summary_grid = ttk.Frame(summary_frame)
        summary_grid.pack(fill=tk.X, padx=5, pady=5)

        # Key metrics
        self.total_value_label = ttk.Label(
            summary_grid, text="Total Value: $0", style="Title.TLabel"
        )
        self.total_value_label.grid(row=0, column=0, sticky=tk.W, padx=10, pady=2)

        self.daily_pnl_label = ttk.Label(summary_grid, text="Daily P&L: $0")
        self.daily_pnl_label.grid(row=0, column=1, sticky=tk.W, padx=10, pady=2)

        self.open_positions_label = ttk.Label(summary_grid, text="Open Positions: 0")
        self.open_positions_label.grid(row=1, column=0, sticky=tk.W, padx=10, pady=2)

        self.cash_label = ttk.Label(summary_grid, text="Available Cash: $0")
        self.cash_label.grid(row=1, column=1, sticky=tk.W, padx=10, pady=2)

        # Positions table
        positions_frame = ttk.LabelFrame(self.portfolio_frame, text="Current Positions")
        positions_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.positions_tree = ttk.Treeview(
            positions_frame,
            columns=("Qty", "Avg_Price", "Current", "PnL", "PnL_Pct", "Value"),
            show="tree headings",
        )
        self.positions_tree.heading("#0", text="Symbol")
        self.positions_tree.heading("Qty", text="Quantity")
        self.positions_tree.heading("Avg_Price", text="Avg Price")
        self.positions_tree.heading("Current", text="Current Price")
        self.positions_tree.heading("PnL", text="P&L")
        self.positions_tree.heading("PnL_Pct", text="P&L %")
        self.positions_tree.heading("Value", text="Market Value")

        # Add scrollbar to positions
        pos_scrollbar = ttk.Scrollbar(
            positions_frame, orient="vertical", command=self.positions_tree.yview
        )
        self.positions_tree.configure(yscrollcommand=pos_scrollbar.set)

        self.positions_tree.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        pos_scrollbar.pack(side="right", fill="y")

    def create_analytics_tab(self):
        """Analytics and charting tab"""
        self.analytics_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.analytics_frame, text="📈 Analytics")

        # Chart controls
        chart_controls = ttk.LabelFrame(self.analytics_frame, text="Chart Controls")
        chart_controls.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(
            chart_controls, text="📊 Equity Curve", command=self.plot_equity_curve
        ).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(
            chart_controls,
            text="📈 Performance Metrics",
            command=self.plot_performance_metrics,
        ).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(
            chart_controls, text="🎯 Trade Analysis", command=self.plot_trade_analysis
        ).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(
            chart_controls,
            text="📉 Drawdown Analysis",
            command=self.plot_drawdown_analysis,
        ).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(  # Add this button
            chart_controls,
            text="🔄 Walk-Forward Analysis",
            command=self.plot_wf_analysis,
        ).pack(side=tk.LEFT, padx=5, pady=5)

        # Chart area
        self.chart_frame = ttk.Frame(self.analytics_frame)
        self.chart_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create matplotlib figure
        self.figure = Figure(figsize=(14, 8), dpi=100, facecolor="white")
        self.canvas = FigureCanvasTkAgg(self.figure, self.chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add navigation toolbar
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk

        self.chart_toolbar = NavigationToolbar2Tk(self.canvas, self.chart_frame)
        self.chart_toolbar.update()

    def create_logs_tab(self):
        """Logs and debugging tab"""
        self.logs_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.logs_frame, text="📋 Logs")

        # Log controls
        log_controls = ttk.Frame(self.logs_frame)
        log_controls.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(log_controls, text="🔄 Refresh", command=self.refresh_logs).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(log_controls, text="🗑️ Clear", command=self.clear_logs).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(log_controls, text="💾 Save", command=self.save_logs).pack(
            side=tk.LEFT, padx=5
        )

        # Log level filter
        ttk.Label(log_controls, text="Level:").pack(side=tk.LEFT, padx=10)
        self.log_level_var = tk.StringVar(value="INFO")
        log_level_combo = ttk.Combobox(
            log_controls,
            textvariable=self.log_level_var,
            values=["DEBUG", "INFO", "WARNING", "ERROR"],
            state="readonly",
            width=10,
        )
        log_level_combo.pack(side=tk.LEFT, padx=5)

        # Logs display
        self.logs_text = scrolledtext.ScrolledText(
            self.logs_frame, height=25, state=tk.DISABLED
        )
        self.logs_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_status_bar(self):
        """Enhanced status bar with multiple sections"""
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # Main status
        self.status_bar = ttk.Label(
            status_frame, text="Ready", relief=tk.SUNKEN, anchor=tk.W
        )
        self.status_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Time display
        self.time_label = ttk.Label(status_frame, text="", relief=tk.SUNKEN, width=20)
        self.time_label.pack(side=tk.RIGHT)

        # Market status
        self.market_status_label = ttk.Label(
            status_frame, text="Market: Closed", relief=tk.SUNKEN, width=15
        )
        self.market_status_label.pack(side=tk.RIGHT)

        # Update time every second
        self.update_time()

    def update_time(self):
        """Update time display"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_label.config(text=current_time)

        # Update market status (simplified)
        hour = datetime.now().hour
        if 9 <= hour <= 16:  # Simplified market hours
            self.market_status_label.config(text="Market: Open", foreground="green")
        else:
            self.market_status_label.config(text="Market: Closed", foreground="red")

        self.root.after(1000, self.update_time)

    # IB Connection Management
    def connect_ib(self):
        """Connect to Interactive Brokers"""
        if self.ib_connected:
            messagebox.showinfo("Info", "Already connected to IB")
            return

        self.update_status("Connecting to IB...")
        self.connection_status.config(text="● Connecting...", foreground="orange")

        def connect_worker():
            try:
                self.ib_app = IbApp()
                self.ib_app.connect(
                    self.host_var.get(),
                    int(self.port_var.get()),
                    clientId=int(self.client_id_var.get()),
                )

                self.ib_reader = threading.Thread(target=self.ib_app.run, daemon=True)
                self.ib_reader.start()

                if self.ib_app.connected_evt.wait(timeout=10):
                    self.ib_connected = True
                    self.message_queue.put(
                        ("connection_success", "Connected to IB successfully!")
                    )
                else:
                    self.message_queue.put(("connection_error", "Connection timeout"))

            except Exception as e:
                self.message_queue.put(("connection_error", f"Connection failed: {e}"))

        self.run_in_thread(connect_worker)

    def disconnect_ib(self):
        """Disconnect from Interactive Brokers"""
        if not self.ib_connected:
            messagebox.showinfo("Info", "Not connected to IB")
            return

        try:
            if self.ib_app and self.ib_app.isConnected():
                self.ib_app.disconnect()
            self.ib_connected = False
            self.connection_status.config(
                text="● Disconnected", style="Disconnected.TLabel"
            )
            self.update_status("Disconnected from IB")
            self.log_message("Disconnected from IB")
            messagebox.showinfo("Success", "Disconnected from IB successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Disconnect failed: {e}")

    # Data Management
    def download_data(self):
        """Download historical data for all symbols"""
        if not self.ib_connected:
            messagebox.showwarning("Warning", "Please connect to IB first")
            return

        symbols = self.get_symbols_list()
        if not symbols:
            messagebox.showwarning("Warning", "No symbols selected")
            return

        self.update_status("Downloading data...")
        self.data_progress.config(mode="determinate", maximum=len(symbols), value=0)

        def download_worker():
            try:
                for i, symbol in enumerate(symbols):
                    if not self.ib_connected:
                        break

                    self.message_queue.put(
                        (
                            "data_progress",
                            f"Downloading {symbol} ({i+1}/{len(symbols)})",
                        )
                    )

                    # **FIX: Pass date range from GUI to download function**
                    start_date = self.start_date_var.get().strip()
                    end_date = self.end_date_var.get().strip()
                    
                    # Create a temporary strategy just for data download
                    strategy_params = self.get_strategy_params()
                    strategy = build_strategy(
                        self.strategy_var.get(), **strategy_params
                    )
                    
                    # Download data with custom date range
                    df = download_with_date_range(
                        self.ib_app,
                        symbol,
                        strategy,
                        start_date=start_date if start_date else None,
                        end_date=end_date if end_date else None
                    )

                    if not df.empty:
                        self.data_cache[symbol] = df
                        status = "✓ Complete"
                        bars = len(df)
                        date_range = (
                            f"{df.index.min().date()} to {df.index.max().date()}"
                        )
                    else:
                        status = "✗ Failed"
                        bars = 0
                        date_range = "N/A"

                    # Update data status table
                    self.message_queue.put(
                        (
                            "data_status_update",
                            {
                                "symbol": symbol,
                                "status": status,
                                "last_updated": datetime.now().strftime(
                                    "%Y-%m-%d %H:%M"
                                ),
                                "bars": bars,
                                "date_range": date_range,
                            },
                        )
                    )

                    self.data_progress.config(value=i + 1)

                self.message_queue.put(
                    (
                        "data_download_complete",
                        f"Downloaded data for {len([s for s in symbols if s in self.data_cache])} symbols",
                    )
                )

            except Exception as e:
                self.message_queue.put(("error", f"Data download failed: {e}"))

        self.run_in_thread(download_worker)

    def update_data(self):
        """Update existing data (delta download)"""
        if not self.ib_connected:
            messagebox.showwarning("Warning", "Please connect to IB first")
            return

        symbols = self.get_symbols_list()
        symbols_to_update = [s for s in symbols if s in self.data_cache]

        if not symbols_to_update:
            messagebox.showinfo("Info", "No cached data to update")
            return

        self.update_status("Updating data...")

        def update_worker():
            try:
                for symbol in symbols_to_update:
                    # This would implement delta update logic
                    # For now, just refresh the data
                    df = latest_completed_daily_df(
                        self.ib_app,
                        symbol,
                        build_strategy(
                            self.strategy_var.get(), **self.get_strategy_params()
                        ),
                    )
                    if not df.empty:
                        self.data_cache[symbol] = df

                self.message_queue.put(
                    ("success", f"Updated data for {len(symbols_to_update)} symbols")
                )
            except Exception as e:
                self.message_queue.put(("error", f"Data update failed: {e}"))

        self.run_in_thread(update_worker)

    # Optimization Methods
    def create_optimization_params(self):
        """Create optimization parameter range inputs"""
        # Clear existing widgets
        for widget in self.param_ranges_frame.winfo_children():
            widget.destroy()
        self.param_range_vars.clear()

        strategy = self.opt_strategy_var.get()
        if strategy not in STRATEGY_PARAM_KEYS:
            return

        params = STRATEGY_PARAM_KEYS[strategy]

        # Default ranges for common parameters
        default_ranges = {
            "len_channel": "15,20,25",
            "adx_len": "10,14,20",
            "adx_thresh": "15,20,25,30",
            "rsi_len": "10,14,20",
            "rsi_thresh": "50,55,60,65",
            "trade_pct": "10,15,20",
            "max_positions": "5,8,10,15",
        }

        row = 0
        for param in sorted(params):
            ttk.Label(
                self.param_ranges_frame, text=f"{param.replace('_', ' ').title()}:"
            ).grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)

            var = tk.StringVar(value=default_ranges.get(param, ""))
            self.param_range_vars[param] = var
            entry = ttk.Entry(self.param_ranges_frame, textvariable=var, width=30)
            entry.grid(row=row, column=1, padx=5, pady=2)

            ttk.Label(self.param_ranges_frame, text="(comma-separated values)").grid(
                row=row, column=2, sticky=tk.W, padx=5, pady=2
            )

            row += 1

    def on_opt_strategy_change(self, event=None):
        """Handle optimization strategy change"""
        self.create_optimization_params()

    def start_optimization(self):
        """Start parameter optimization"""
        if not self.data_cache:
            messagebox.showwarning("Warning", "No data available. Download data first.")
            return

        strategy = self.opt_strategy_var.get()
        if not strategy:
            messagebox.showwarning("Warning", "Please select a strategy to optimize")
            return

        # Parse parameter ranges
        param_ranges = {}
        for param, var in self.param_range_vars.items():
            range_str = var.get().strip()
            if range_str:
                try:
                    values = [
                        float(x.strip()) if "." in x.strip() else int(x.strip())
                        for x in range_str.split(",")
                    ]
                    param_ranges[param] = values
                except ValueError:
                    messagebox.showerror(
                        "Error", f"Invalid range for {param}: {range_str}"
                    )
                    return

        if not param_ranges:
            messagebox.showwarning("Warning", "Please specify parameter ranges")
            return

        self.update_status("Starting optimization...")
        self.opt_progress.config(mode="determinate", maximum=100, value=0)

        def optimization_callback(message, msg_type):
            if msg_type == "progress":
                # Extract progress percentage
                if "%" in message:
                    try:
                        progress = float(message.split(":")[1].split("%")[0].strip())
                        self.opt_progress.config(value=progress)
                    except:
                        pass
            self.message_queue.put((msg_type, message))

        def run_opt():
            self.optimizer.run_optimization(
                symbols=list(self.data_cache.keys()),
                strategy_name=strategy,
                param_ranges=param_ranges,
                data_dict=self.data_cache,
                callback=optimization_callback,
            )

            # Display results
            best_results = self.optimizer.get_best_results(
                metric=self.opt_metric_var.get(), top_n=20
            )
            self.message_queue.put(("optimization_results", best_results))

        self.run_in_thread(run_opt)

    def stop_optimization(self):
        """Stop running optimization"""
        self.optimizer.stop_optimization()
        self.update_status("Optimization stopped")

    def display_optimization_results(self, results):
        """Display optimization results"""
        self.opt_results_tree.delete(*self.opt_results_tree.get_children())

        for i, result in enumerate(results, 1):
            params_str = ", ".join([f"{k}={v}" for k, v in result["params"].items()])

            self.opt_results_tree.insert(
                "",
                "end",
                text=str(i),
                values=(
                    f"{result['total_return']:.2f}",
                    f"{result['sharpe']:.3f}" if result["sharpe"] != "NA" else "NA",
                    f"{result['max_dd']:.2f}",
                    (
                        f"{result['profit_factor']:.2f}"
                        if result["profit_factor"] != "Inf"
                        else "Inf"
                    ),
                    result["trades_count"],
                    params_str,
                ),
            )

    def export_optimization_results(self):
        """Export optimization results"""
        if not self.optimizer.results:
            messagebox.showwarning("Warning", "No optimization results to export")
            return

        filename = filedialog.asksaveasfilename(
            title="Export Optimization Results",
            defaultextension=".csv",
            filetypes=[
                ("CSV files", "*.csv"),
                ("Excel files", "*.xlsx"),
                ("All files", "*.*"),
            ],
        )

        if filename:
            try:
                # Convert results to DataFrame
                data = []
                for result in self.optimizer.results:
                    row = result["params"].copy()
                    row.update(
                        {
                            "total_return": result["total_return"],
                            "sharpe": result["sharpe"],
                            "max_dd": result["max_dd"],
                            "profit_factor": result["profit_factor"],
                            "trades_count": result["trades_count"],
                        }
                    )
                    data.append(row)

                df = pd.DataFrame(data)

                if filename.endswith(".csv"):
                    df.to_csv(filename, index=False)
                elif filename.endswith(".xlsx"):
                    df.to_excel(filename, index=False)
                else:
                    df.to_csv(filename, index=False)

                messagebox.showinfo(
                    "Success", f"Optimization results exported to {filename}"
                )
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {e}")

    # Enhanced Analytics and Charting
    def plot_equity_curve(self):
        """Plot equity curve from backtest results"""
        if not self.backtest_results:
            messagebox.showwarning(
                "Warning", "No backtest results available. Run a backtest first."
            )
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # Get actual equity data if available
        equity_file = os.path.join(self.output_dir_var.get(), "adx_equity.csv")
        if os.path.exists(equity_file):
            try:
                equity_df = pd.read_csv(equity_file)
                equity_df["date"] = pd.to_datetime(equity_df["date"])

                ax.plot(
                    equity_df["date"],
                    equity_df["equity"],
                    "b-",
                    linewidth=2,
                    label="Portfolio Value",
                )
                ax.set_title("Portfolio Equity Curve", fontsize=14, fontweight="bold")
                ax.set_xlabel("Date")
                ax.set_ylabel("Portfolio Value ($)")
                ax.grid(True, alpha=0.3)
                ax.legend()

                # Format y-axis as currency
                ax.yaxis.set_major_formatter(
                    plt.FuncFormatter(lambda x, p: f"${x:,.0f}")
                )

                # Rotate x-axis labels for better readability
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

            except Exception as e:
                self.log_message(f"Error plotting equity curve: {e}", "ERROR")
                return
        else:
            # Generate sample data for demonstration
            dates = pd.date_range("2020-01-01", periods=252, freq="D")
            returns = np.random.normal(0.0008, 0.02, 252)  # Daily returns
            equity = 10000 * (1 + returns).cumprod()

            ax.plot(dates, equity, "b-", linewidth=2, label="Portfolio Value")
            ax.set_title("Sample Equity Curve", fontsize=14, fontweight="bold")
            ax.set_xlabel("Date")
            ax.set_ylabel("Portfolio Value ($)")
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

        self.figure.tight_layout()
        self.canvas.draw()

    def plot_performance_metrics(self):
        """Plot performance metrics"""
        if not self.backtest_results:
            messagebox.showwarning(
                "Warning", "No backtest results available. Run a backtest first."
            )
            return

        self.figure.clear()

        # Get performance metrics
        summary = self.backtest_results.get("summary", {})

        # Create subplots for different metric categories
        fig = self.figure

        # Returns metrics
        ax1 = fig.add_subplot(2, 2, 1)
        return_metrics = ["TotalReturnPct", "CAGR", "ROR"]
        return_values = [float(summary.get(m, 0)) for m in return_metrics]
        return_labels = ["Total Return", "CAGR", "ROR"]

        bars1 = ax1.bar(
            return_labels, return_values, color=["green", "blue", "teal"], alpha=0.7
        )
        ax1.set_title("Return Metrics (%)", fontweight="bold")
        ax1.set_ylabel("Percentage")
        for bar, value in zip(bars1, return_values):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.5,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
            )

        # Risk metrics
        ax2 = fig.add_subplot(2, 2, 2)
        risk_metrics = ["MaxDDPct", "AnnVolPct"]
        risk_values = []
        for m in risk_metrics:
            val = summary.get(m, 0)
            if val == "NA":
                val = 0
            risk_values.append(float(val))
        risk_labels = ["Max Drawdown", "Annual Volatility"]

        bars2 = ax2.bar(risk_labels, risk_values, color=["red", "orange"], alpha=0.7)
        ax2.set_title("Risk Metrics (%)", fontweight="bold")
        ax2.set_ylabel("Percentage")
        for bar, value in zip(bars2, risk_values):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.2,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
            )

        # Risk-adjusted metrics
        ax3 = fig.add_subplot(2, 2, 3)
        ratio_metrics = ["Sharpe", "Calmar"]
        ratio_values = []
        for m in ratio_metrics:
            val = summary.get(m, 0)
            if val == "NA":
                val = 0
            ratio_values.append(float(val))
        ratio_labels = ["Sharpe Ratio", "Calmar Ratio"]

        bars3 = ax3.bar(
            ratio_labels, ratio_values, color=["purple", "brown"], alpha=0.7
        )
        ax3.set_title("Risk-Adjusted Returns", fontweight="bold")
        ax3.set_ylabel("Ratio")
        for bar, value in zip(bars3, ratio_values):
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.02,
                f"{value:.2f}",
                ha="center",
                va="bottom",
            )

        # Trading metrics
        ax4 = fig.add_subplot(2, 2, 4)
        trade_metrics = ["Trades", "PctWins"]
        trade_values = [int(summary.get("Trades", 0)), float(summary.get("PctWins", 0))]
        trade_labels = ["Total Trades", "Win Rate (%)"]

        bars4 = ax4.bar(
            trade_labels, trade_values, color=["gold", "lightgreen"], alpha=0.7
        )
        ax4.set_title("Trading Statistics", fontweight="bold")
        for i, (bar, value) in enumerate(zip(bars4, trade_values)):
            height = bar.get_height()
            if i == 0:  # Trades count
                ax4.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.5,
                    f"{int(value)}",
                    ha="center",
                    va="bottom",
                )
            else:  # Win rate
                ax4.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 1,
                    f"{value:.1f}%",
                    ha="center",
                    va="bottom",
                )

        fig.tight_layout()
        self.canvas.draw()

    def plot_trade_analysis(self):
        """Plot comprehensive trade analysis"""
        trades_file = os.path.join(self.output_dir_var.get(), "adx_trades.csv")

        if not os.path.exists(trades_file):
            messagebox.showwarning(
                "Warning", "No trades data available. Run a backtest first."
            )
            return

        try:
            trades_df = pd.read_csv(trades_file)
            if trades_df.empty:
                messagebox.showinfo("Info", "No trades in the dataset")
                return

        except Exception as e:
            self.log_message(f"Error loading trades data: {e}", "ERROR")
            return

        self.figure.clear()
        fig = self.figure

        # Convert return percentage to decimal for calculations
        returns = trades_df["ret_pct"] / 100.0

        # Plot 1: Trade return distribution
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.hist(returns, bins=20, alpha=0.7, color="blue", edgecolor="black")
        ax1.axvline(
            returns.mean(),
            color="red",
            linestyle="--",
            label=f"Mean: {returns.mean():.3f}",
        )
        ax1.set_title("Trade Return Distribution", fontweight="bold")
        ax1.set_xlabel("Return")
        ax1.set_ylabel("Frequency")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Cumulative returns
        ax2 = fig.add_subplot(2, 2, 2)
        cumulative_returns = (1 + returns).cumprod()
        ax2.plot(cumulative_returns, "g-", linewidth=2)
        ax2.set_title("Cumulative Trade Returns", fontweight="bold")
        ax2.set_xlabel("Trade Number")
        ax2.set_ylabel("Cumulative Return")
        ax2.grid(True, alpha=0.3)

        # Plot 3: Monthly win/loss analysis
        ax3 = fig.add_subplot(2, 2, 3)
        trades_df["entry_date"] = pd.to_datetime(trades_df["entry_date"])
        trades_df["month"] = trades_df["entry_date"].dt.to_period("M")

        monthly_stats = (
            trades_df.groupby("month")
            .agg(
                {"ret_pct": ["count", lambda x: (x > 0).sum(), lambda x: (x < 0).sum()]}
            )
            .round(2)
        )

        if len(monthly_stats) > 0:
            months = [str(m) for m in monthly_stats.index[-6:]]  # Last 6 months
            wins = monthly_stats["ret_pct"]["<lambda_0>"].values[-6:]
            losses = monthly_stats["ret_pct"]["<lambda_1>"].values[-6:]

            x = np.arange(len(months))
            width = 0.35

            ax3.bar(x - width / 2, wins, width, label="Wins", color="green", alpha=0.7)
            ax3.bar(
                x + width / 2, losses, width, label="Losses", color="red", alpha=0.7
            )
            ax3.set_title("Win/Loss by Month", fontweight="bold")
            ax3.set_xlabel("Month")
            ax3.set_ylabel("Number of Trades")
            ax3.set_xticks(x)
            ax3.set_xticklabels(months, rotation=45)
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # Plot 4: Drawdown analysis
        ax4 = fig.add_subplot(2, 2, 4)
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max

        ax4.fill_between(
            range(len(drawdown)), drawdown, 0, alpha=0.3, color="red", label="Drawdown"
        )
        ax4.plot(drawdown, "r-", linewidth=1)
        ax4.set_title("Trade-Level Drawdown", fontweight="bold")
        ax4.set_xlabel("Trade Number")
        ax4.set_ylabel("Drawdown")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        fig.tight_layout()
        self.canvas.draw()

    def plot_drawdown_analysis(self):
        """Plot detailed drawdown analysis"""
        equity_file = os.path.join(self.output_dir_var.get(), "adx_equity.csv")

        if not os.path.exists(equity_file):
            messagebox.showwarning(
                "Warning", "No equity data available. Run a backtest first."
            )
            return

        try:
            equity_df = pd.read_csv(equity_file)
            equity_df["date"] = pd.to_datetime(equity_df["date"])

        except Exception as e:
            self.log_message(f"Error loading equity data: {e}", "ERROR")
            return

        self.figure.clear()
        fig = self.figure

        equity = equity_df["equity"]
        dates = equity_df["date"]

        # Calculate drawdown
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max * 100

        # Plot 1: Equity curve with drawdown
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.plot(dates, equity, "b-", linewidth=2, label="Portfolio Value")
        ax1.plot(
            dates, running_max, "g--", linewidth=1, label="Running Maximum", alpha=0.7
        )
        ax1.set_title("Portfolio Value vs Running Maximum", fontweight="bold")
        ax1.set_ylabel("Portfolio Value ($)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

        # Plot 2: Drawdown periods
        ax2 = fig.add_subplot(2, 1, 2)
        ax2.fill_between(dates, drawdown, 0, alpha=0.3, color="red", label="Drawdown")
        ax2.plot(dates, drawdown, "r-", linewidth=1)
        ax2.set_title("Drawdown Analysis", fontweight="bold")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Drawdown (%)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Add maximum drawdown line
        max_dd = drawdown.min()
        ax2.axhline(
            max_dd,
            color="darkred",
            linestyle="--",
            label=f"Max Drawdown: {max_dd:.2f}%",
        )
        ax2.legend()

        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

        fig.tight_layout()
        self.canvas.draw()

    def plot_wf_analysis(self):
        """Plot walk-forward analysis results"""
        if not hasattr(self.walkforward, 'results') or not self.walkforward.results:
            messagebox.showwarning("Warning", "No walk-forward results available")
            return

        self.figure.clear()
        fig = self.figure

        # Extract data for plotting
        windows = []
        oos_returns = []
        train_returns = []
        
        for result in self.walkforward.results:
            windows.append(result.get("window", 0))
            oos_results = result.get("oos_results", {})
            oos_return = oos_results.get("TotalReturnPct", 0)
            
            if isinstance(oos_return, (int, float)):
                oos_returns.append(oos_return)
            else:
                oos_returns.append(0)

        if not oos_returns:
            messagebox.showinfo("Info", "No valid walk-forward data to plot")
            return

        # Plot 1: OOS Returns by Window
        ax1 = fig.add_subplot(2, 2, 1)
        bars = ax1.bar(windows, oos_returns, alpha=0.7, 
                    color=['green' if x > 0 else 'red' for x in oos_returns])
        ax1.set_title("Out-of-Sample Returns by Window", fontweight="bold")
        ax1.set_xlabel("Window")
        ax1.set_ylabel("Return (%)")
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)

        # Plot 2: Cumulative OOS Performance
        ax2 = fig.add_subplot(2, 2, 2)
        cumulative_returns = np.cumprod([1 + r/100 for r in oos_returns])
        ax2.plot(windows, cumulative_returns, 'b-', linewidth=2, marker='o')
        ax2.set_title("Cumulative Out-of-Sample Performance", fontweight="bold")
        ax2.set_xlabel("Window")
        ax2.set_ylabel("Cumulative Return")
        ax2.grid(True, alpha=0.3)

        # Plot 3: Distribution of OOS Returns
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.hist(oos_returns, bins=min(10, len(oos_returns)), alpha=0.7, color="blue", edgecolor="black")
        ax3.axvline(np.mean(oos_returns), color="red", linestyle="--", 
                    label=f"Mean: {np.mean(oos_returns):.2f}%")
        ax3.set_title("Distribution of OOS Returns", fontweight="bold")
        ax3.set_xlabel("Return (%)")
        ax3.set_ylabel("Frequency")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Rolling Statistics
        if len(oos_returns) >= 3:
            ax4 = fig.add_subplot(2, 2, 4)
            
            # Calculate rolling statistics
            window_size = min(3, len(oos_returns))
            rolling_mean = pd.Series(oos_returns).rolling(window=window_size).mean()
            rolling_std = pd.Series(oos_returns).rolling(window=window_size).std()
            
            ax4.plot(windows, rolling_mean, 'g-', linewidth=2, label=f'Rolling Mean ({window_size})')
            ax4.fill_between(windows, 
                            rolling_mean - rolling_std, 
                            rolling_mean + rolling_std, 
                            alpha=0.3, color='green', label='±1 Std Dev')
            ax4.set_title("Rolling Statistics", fontweight="bold")
            ax4.set_xlabel("Window")
            ax4.set_ylabel("Return (%)")
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            # Not enough data for rolling stats
            ax4 = fig.add_subplot(2, 2, 4)
            ax4.text(0.5, 0.5, 'Insufficient data\nfor rolling statistics', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title("Rolling Statistics", fontweight="bold")

        fig.tight_layout()
        self.canvas.draw()

    # Configuration and utility methods
    def load_saved_config(self):
        """Load configuration from file"""
        self.host_var.set(self.config.get("CONNECTION", "host", "127.0.0.1"))
        self.port_var.set(self.config.get("CONNECTION", "port", "7497"))
        self.client_id_var.set(self.config.get("CONNECTION", "client_id", "44"))

        self.start_date_var.set(self.config.get("DATA", "start_date", "2020-01-01"))
        self.end_date_var.set(self.config.get("DATA", "end_date", ""))
        self.timeframe_var.set(self.config.get("DATA", "timeframe", "1 day"))
        self.data_type_var.set(self.config.get("DATA", "data_type", "TRADES"))

        self.strategy_var.set(self.config.get("STRATEGY", "current", "adx_squeeze"))
        self.opt_strategy_var.set(self.config.get("STRATEGY", "current", "adx_squeeze"))

        self.output_dir_var.set(self.config.get("GENERAL", "output_dir", "output"))
        self.equity_var.set(self.config.get("GENERAL", "equity", "10000"))

        self.on_strategy_change()
        self.on_opt_strategy_change()
        self.use_default_symbols()

    def save_config_manually(self):
        """Save current configuration"""
        self.config.set("CONNECTION", "host", self.host_var.get())
        self.config.set("CONNECTION", "port", self.port_var.get())
        self.config.set("CONNECTION", "client_id", self.client_id_var.get())

        self.config.set("DATA", "start_date", self.start_date_var.get())
        self.config.set("DATA", "end_date", self.end_date_var.get())
        self.config.set("DATA", "timeframe", self.timeframe_var.get())
        self.config.set("DATA", "data_type", self.data_type_var.get())

        self.config.set("STRATEGY", "current", self.strategy_var.get())
        self.config.set("GENERAL", "output_dir", self.output_dir_var.get())
        self.config.set("GENERAL", "equity", self.equity_var.get())

        # Save strategy parameters
        for param, var in self.param_vars.items():
            self.config.set("STRATEGY", param, var.get())

        self.config.save_config()
        messagebox.showinfo("Success", "Configuration saved successfully!")

    def create_strategy_params(self):
        """Create strategy parameter inputs"""
        # Clear existing parameters
        for widget in self.params_frame.winfo_children():
            widget.destroy()
        self.param_vars.clear()

        strategy = self.strategy_var.get()
        if strategy not in STRATEGY_PARAM_KEYS:
            return

        params = STRATEGY_PARAM_KEYS[strategy]

        # Default values
        defaults = {
            "len_channel": self.config.get("STRATEGY", "len_channel", "20"),
            "adx_len": self.config.get("STRATEGY", "adx_len", "15"),
            "adx_thresh": self.config.get("STRATEGY", "adx_thresh", "20.0"),
            "rsi_len": self.config.get("STRATEGY", "rsi_len", "14"),
            "rsi_thresh": self.config.get("STRATEGY", "rsi_thresh", "55.0"),
            "trade_pct": self.config.get("STRATEGY", "trade_pct", "15.0"),
            "max_positions": self.config.get("STRATEGY", "max_positions", "10"),
            "max_exposure_pct": self.config.get(
                "STRATEGY", "max_exposure_pct", "100.0"
            ),
            "warmup_bars": self.config.get("STRATEGY", "warmup_bars", "80"),
        }

        row = 0
        for param in sorted(params):
            ttk.Label(
                self.params_frame, text=f"{param.replace('_', ' ').title()}:"
            ).grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)

            var = tk.StringVar(value=defaults.get(param, ""))
            self.param_vars[param] = var
            entry = ttk.Entry(self.params_frame, textvariable=var, width=15)
            entry.grid(row=row, column=1, padx=5, pady=2)

            row += 1

    def on_strategy_change(self, event=None):
        """Handle strategy selection change"""
        self.create_strategy_params()

    def update_symbol_count(self, event=None):
        """Update symbol count display"""
        symbols = self.get_symbols_list()
        self.symbol_count_label.config(text=f"Symbols: {len(symbols)}")

    # File operations
    def load_symbols_file(self):
        """Load symbols from file"""
        filename = filedialog.askopenfilename(
            title="Load Symbols File",
            filetypes=[
                ("Text files", "*.txt"),
                ("CSV files", "*.csv"),
                ("All files", "*.*"),
            ],
        )
        if filename:
            try:
                if filename.endswith(".csv"):
                    df = pd.read_csv(filename)
                    if "symbol" in df.columns:
                        symbols = df["symbol"].tolist()
                    else:
                        symbols = df.iloc[:, 0].tolist()
                else:
                    with open(filename, "r") as f:
                        symbols = [
                            line.strip()
                            for line in f
                            if line.strip() and not line.startswith("#")
                        ]

                self.symbols_text.delete(1.0, tk.END)
                self.symbols_text.insert(1.0, "\n".join(symbols))
                self.update_symbol_count()
                self.log_message(f"Loaded {len(symbols)} symbols from {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {e}")

    def save_symbols_file(self):
        """Save current symbols to file"""
        filename = filedialog.asksaveasfilename(
            title="Save Symbols File",
            defaultextension=".txt",
            filetypes=[
                ("Text files", "*.txt"),
                ("CSV files", "*.csv"),
                ("All files", "*.*"),
            ],
        )
        if filename:
            try:
                symbols = self.get_symbols_list()
                if filename.endswith(".csv"):
                    df = pd.DataFrame({"symbol": symbols})
                    df.to_csv(filename, index=False)
                else:
                    with open(filename, "w") as f:
                        f.write("\n".join(symbols))
                messagebox.showinfo(
                    "Success", f"Saved {len(symbols)} symbols to {filename}"
                )
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file: {e}")

    def use_default_symbols(self):
        """Load default symbols"""
        default_symbols = [
            "MSFT",
            "NVDA",
            "AAPL",
            "AMZN",
            "GOOGL",
            "META",
            "AVGO",
            "TSLA",
            "WMT",
            "JPM",
            "V",
            "SPY",
            "BRK.A",
        ]
        self.symbols_text.delete(1.0, tk.END)
        self.symbols_text.insert(1.0, "\n".join(default_symbols))
        self.update_symbol_count()

    # Utility methods
    def get_symbols_list(self):
        """Get list of symbols from text widget"""
        text = self.symbols_text.get(1.0, tk.END).strip()
        return [s.strip().upper() for s in text.split("\n") if s.strip()]

    def get_strategy_params(self):
        """Get strategy parameters"""
        params = {}
        for param, var in self.param_vars.items():
            value = var.get().strip()
            if value:
                try:
                    if "." in value:
                        params[param] = float(value)
                    else:
                        params[param] = int(value)
                except ValueError:
                    params[param] = value
        return params

    def log_message(self, message, level="INFO"):
        """Add message to logs"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}\n"

        self.logs_text.config(state=tk.NORMAL)
        self.logs_text.insert(tk.END, log_entry)
        self.logs_text.see(tk.END)
        self.logs_text.config(state=tk.DISABLED)

        logger.info(message)

    # Continuation of remaining methods...
    def test_connection(self):
        """Test IB connection"""
        self.update_status("Testing connection...")

        def test_worker():
            try:
                test_app = IbApp()
                test_app.connect(
                    self.host_var.get(),
                    int(self.port_var.get()),
                    clientId=int(self.client_id_var.get()) + 1000,
                )  # Different client ID for testing

                reader = threading.Thread(target=test_app.run, daemon=True)
                reader.start()

                if test_app.connected_evt.wait(timeout=10):
                    self.message_queue.put(
                        ("connection_success", "Connection test successful!")
                    )
                else:
                    self.message_queue.put(
                        ("connection_error", "Connection test timeout")
                    )

                if test_app.isConnected():
                    test_app.disconnect()

            except Exception as e:
                self.message_queue.put(
                    ("connection_error", f"Connection test failed: {e}")
                )

        self.run_in_thread(test_worker)

    def validate_symbols(self):
        """Validate symbols with IB"""
        if not self.ib_connected:
            messagebox.showwarning("Warning", "Please connect to IB first")
            return

        symbols = self.get_symbols_list()
        if not symbols:
            messagebox.showwarning("Warning", "No symbols to validate")
            return

        self.update_status(f"Validating {len(symbols)} symbols...")

        def validate_worker():
            try:
                valid_symbols = []
                invalid_symbols = []

                for symbol in symbols:
                    # Use your existing contract resolution function
                    from ibapi_appv1_patched import resolve_contract

                    contract = resolve_contract(self.ib_app, symbol)

                    if contract:
                        valid_symbols.append(symbol)
                    else:
                        invalid_symbols.append(symbol)

                self.message_queue.put(
                    (
                        "validation_complete",
                        {"valid": valid_symbols, "invalid": invalid_symbols},
                    )
                )

            except Exception as e:
                self.message_queue.put(("error", f"Validation failed: {e}"))

        self.run_in_thread(validate_worker)

    def clear_cache(self):
        """Clear data cache"""
        if messagebox.askyesno("Confirm", "Clear all cached data?"):
            try:
                cache_dir = Path("cache_parquet")
                if cache_dir.exists():
                    import shutil

                    shutil.rmtree(cache_dir)
                    cache_dir.mkdir()

                self.data_cache.clear()
                self.data_status_tree.delete(*self.data_status_tree.get_children())

                self.log_message("Cache cleared successfully")
                messagebox.showinfo("Success", "Cache cleared successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to clear cache: {e}")

    # Trading operations (continued from earlier methods)
    def quick_backtest(self):
        """Run a quick backtest with limited data"""
        if not self.data_cache:
            messagebox.showwarning("Warning", "No data available. Download data first.")
            return

        self.update_status("Running quick backtest...")
        self.backtest_progress.start()

        def quick_backtest_worker():
            try:
                # Use cached data for quick backtest
                strategy_params = self.get_strategy_params()
                strategy = build_strategy(self.strategy_var.get(), **strategy_params)

                # Use only first 5 symbols for quick test
                limited_data = dict(list(self.data_cache.items())[:5])

                trades_df, summary, equity_df = backtest_portfolio(
                    limited_data, strategy
                )

                self.message_queue.put(
                    (
                        "backtest_complete",
                        {
                            "summary": summary,
                            "trades_df": trades_df,
                            "equity_df": equity_df,
                        },
                    )
                )

            except Exception as e:
                self.message_queue.put(("error", f"Quick backtest failed: {e}"))
            finally:
                self.message_queue.put(("done", None))

        self.run_in_thread(quick_backtest_worker)

    def run_backtest(self):
        """Run full backtest"""
        if not self.data_cache:
            messagebox.showwarning("Warning", "No data available. Download data first.")
            return

        self.backtest_progress.start()
        self.backtest_output.delete(1.0, tk.END)
        self.update_status("Running backtest...")
        self.log_message("Full backtest started")

        def backtest_worker():
            try:
                strategy_params = self.get_strategy_params()
                strategy = build_strategy(self.strategy_var.get(), **strategy_params)

                self.message_queue.put(("progress", "Running backtest calculations..."))

                trades_df, summary, equity_df = backtest_portfolio(
                    self.data_cache, strategy
                )

                # Save results
                output_dir = self.output_dir_var.get()
                save_csvs(output_dir, trades_df, summary, equity_df)

                # Generate reports if requested
                if self.html_report_var.get():
                    html_path = _html_report(output_dir, summary, equity_df, trades_df)
                    self.message_queue.put(("info", f"HTML report: {html_path}"))

                if self.json_report_var.get():
                    json_path = _json_report(output_dir, summary)
                    self.message_queue.put(("info", f"JSON report: {json_path}"))

                self.message_queue.put(
                    (
                        "backtest_complete",
                        {
                            "summary": summary,
                            "trades_df": trades_df,
                            "equity_df": equity_df,
                        },
                    )
                )

                if self.auto_open_var.get():
                    self.message_queue.put(("auto_open", None))

            except Exception as e:
                self.message_queue.put(
                    ("error", f"Backtest failed: {str(e)}\n{traceback.format_exc()}")
                )
            finally:
                self.message_queue.put(("done", None))

        self.run_in_thread(backtest_worker)

    def place_orders(self):
        """Place paper trading orders"""
        if not self.ib_connected:
            messagebox.showwarning("Warning", "Please connect to IB first")
            return

        self.trading_output.delete(1.0, tk.END)
        self.update_status("Placing orders...")
        self.log_message("Placing paper trading orders")

        def place_orders_worker():
            try:
                symbols = self.get_symbols_list()
                strategy_params = self.get_strategy_params()
                strategy = build_strategy(self.strategy_var.get(), **strategy_params)
                equity = float(self.equity_var.get())

                import io

                old_stdout = sys.stdout
                sys.stdout = mystdout = io.StringIO()

                try:
                    place_paper_orders_now(
                        symbols=symbols,
                        strategy=strategy,
                        host=self.host_var.get(),
                        port=int(self.port_var.get()),
                        client_id=int(self.client_id_var.get()),
                        equity=equity,
                    )
                    self.message_queue.put(("success", "Orders placed successfully!"))
                finally:
                    sys.stdout = old_stdout
                    output = mystdout.getvalue()
                    if output:
                        self.message_queue.put(("trading_output", output))

            except Exception as e:
                self.message_queue.put(("error", f"Order placement failed: {str(e)}"))

        self.run_in_thread(place_orders_worker)

    def roll_orders(self):
        """Roll orders after market close"""
        if not self.ib_connected:
            messagebox.showwarning("Warning", "Please connect to IB first")
            return

        self.trading_output.delete(1.0, tk.END)
        self.update_status("Rolling orders...")
        self.log_message("Rolling orders after market close")

        def roll_orders_worker():
            try:
                symbols = self.get_symbols_list()
                strategy_params = self.get_strategy_params()
                strategy = build_strategy(self.strategy_var.get(), **strategy_params)
                equity = float(self.equity_var.get())

                import io

                old_stdout = sys.stdout
                sys.stdout = mystdout = io.StringIO()

                try:
                    roll_daily_brackets_after_close(
                        symbols=symbols,
                        strategy=strategy,
                        host=self.host_var.get(),
                        port=int(self.port_var.get()),
                        client_id=int(self.client_id_var.get()),
                        equity=equity,
                    )
                    self.message_queue.put(("success", "Orders rolled successfully!"))
                finally:
                    sys.stdout = old_stdout
                    output = mystdout.getvalue()
                    if output:
                        self.message_queue.put(("trading_output", output))

            except Exception as e:
                self.message_queue.put(("error", f"Order rolling failed: {str(e)}"))

        self.run_in_thread(roll_orders_worker)

    def preview_orders(self):
        """Preview what orders would be placed"""
        self.trading_output.delete(1.0, tk.END)
        self.trading_output.insert(
            tk.END, "Preview mode - no actual orders will be placed\n\n"
        )
        self.update_status("Preview mode")
        self.log_message("Preview mode activated")

    def show_current_signals(self):
        """Show current trading signals"""
        if not self.data_cache:
            messagebox.showwarning("Warning", "No data available. Download data first.")
            return

        self.update_status("Analyzing current signals...")

        def signals_worker():
            try:
                symbols = self.get_symbols_list()
                strategy_params = self.get_strategy_params()
                strategy = build_strategy(self.strategy_var.get(), **strategy_params)
                equity = float(self.equity_var.get())

                signals = []

                for symbol in symbols:
                    if symbol in self.data_cache:
                        df = self.data_cache[symbol]
                        if len(df) < strategy.warmup_bars:
                            continue

                        last_row = df.iloc[-1]

                        # Check if eligible and get entry spec
                        if strategy.is_eligible(last_row):
                            entry_spec = strategy.next_entry_spec(symbol, last_row)
                            if entry_spec:
                                entry_price, stop_loss = entry_spec
                                quantity = strategy.shares_for_entry(
                                    entry_price, equity
                                )
                                risk = (entry_price - stop_loss) * quantity

                                signals.append(
                                    {
                                        "symbol": symbol,
                                        "entry_price": entry_price,
                                        "stop_loss": stop_loss,
                                        "quantity": quantity,
                                        "risk": risk,
                                        "eligible": True,
                                    }
                                )
                        else:
                            signals.append(
                                {
                                    "symbol": symbol,
                                    "entry_price": 0,
                                    "stop_loss": 0,
                                    "quantity": 0,
                                    "risk": 0,
                                    "eligible": False,
                                }
                            )

                self.message_queue.put(("signals_update", signals))

            except Exception as e:
                self.message_queue.put(("error", f"Signal analysis failed: {e}"))

        self.run_in_thread(signals_worker)

    # File and view operations
    def browse_output_dir(self):
        """Browse for output directory"""
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir_var.set(directory)

    def open_output_folder(self):
        """Open the output directory in file explorer"""
        output_dir = self.output_dir_var.get()
        if os.path.exists(output_dir):
            if sys.platform.startswith("win"):
                os.startfile(output_dir)
            elif sys.platform.startswith("darwin"):
                subprocess.run(["open", output_dir])
            else:
                subprocess.run(["xdg-open", output_dir])
        else:
            messagebox.showwarning(
                "Warning", f"Output directory does not exist: {output_dir}"
            )

    def view_html_report(self):
        """Open HTML report in browser"""
        report_path = os.path.join(self.output_dir_var.get(), "report.html")
        if os.path.exists(report_path):
            webbrowser.open(f"file://{os.path.abspath(report_path)}")
        else:
            messagebox.showwarning(
                "Warning", "HTML report not found. Run a backtest first."
            )

    def export_results(self):
        """Export results to various formats"""
        if not self.backtest_results:
            messagebox.showwarning(
                "Warning", "No results to export. Run a backtest first."
            )
            return

        filename = filedialog.asksaveasfilename(
            title="Export Results",
            defaultextension=".xlsx",
            filetypes=[
                ("Excel files", "*.xlsx"),
                ("CSV files", "*.csv"),
                ("JSON files", "*.json"),
                ("All files", "*.*"),
            ],
        )

        if filename:
            try:
                summary = self.backtest_results.get("summary", {})

                if filename.endswith(".xlsx"):
                    with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
                        # Summary sheet
                        summary_df = pd.DataFrame([summary])
                        summary_df.to_excel(writer, sheet_name="Summary", index=False)

                        # Trades sheet if available
                        trades_file = os.path.join(
                            self.output_dir_var.get(), "adx_trades.csv"
                        )
                        if os.path.exists(trades_file):
                            trades_df = pd.read_csv(trades_file)
                            trades_df.to_excel(writer, sheet_name="Trades", index=False)

                        # Equity sheet if available
                        equity_file = os.path.join(
                            self.output_dir_var.get(), "adx_equity.csv"
                        )
                        if os.path.exists(equity_file):
                            equity_df = pd.read_csv(equity_file)
                            equity_df.to_excel(writer, sheet_name="Equity", index=False)

                elif filename.endswith(".json"):
                    with open(filename, "w") as f:
                        json.dump(self.backtest_results, f, indent=2, default=str)

                else:  # CSV
                    summary_df = pd.DataFrame([summary])
                    summary_df.to_csv(filename, index=False)

                messagebox.showinfo("Success", f"Results exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {e}")

    # Menu operations
    def load_config_file(self):
        """Load configuration from file"""
        filename = filedialog.askopenfilename(
            title="Load Configuration",
            filetypes=[("INI files", "*.ini"), ("All files", "*.*")],
        )
        if filename:
            try:
                self.config.config.read(filename)
                self.load_saved_config()
                messagebox.showinfo("Success", "Configuration loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load configuration: {e}")

    def save_config_file(self):
        """Save configuration to file"""
        filename = filedialog.asksaveasfilename(
            title="Save Configuration",
            defaultextension=".ini",
            filetypes=[("INI files", "*.ini"), ("All files", "*.*")],
        )
        if filename:
            try:
                self.save_config_manually()
                with open(filename, "w") as f:
                    self.config.config.write(f)
                messagebox.showinfo("Success", f"Configuration saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save configuration: {e}")

    def compare_strategies(self):
        """Compare multiple strategies"""
        if not self.data_cache:
            messagebox.showwarning("Warning", "No data available for comparison")
            return

        # Create strategy comparison dialog
        comparison_window = tk.Toplevel(self.root)
        comparison_window.title("Strategy Comparison")
        comparison_window.geometry("800x600")

        # Strategy selection
        strategies_frame = ttk.LabelFrame(comparison_window, text="Select Strategies")
        strategies_frame.pack(fill=tk.X, padx=10, pady=10)

        strategy_vars = {}
        for i, strategy in enumerate(STRATEGY_REGISTRY.keys()):
            var = tk.BooleanVar()
            strategy_vars[strategy] = var
            ttk.Checkbutton(strategies_frame, text=strategy, variable=var).grid(
                row=i // 3, column=i % 3, sticky=tk.W, padx=10, pady=5
            )

        # Results display
        results_frame = ttk.LabelFrame(comparison_window, text="Comparison Results")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        comparison_tree = ttk.Treeview(
            results_frame,
            columns=("TotalReturn", "Sharpe", "MaxDD", "Trades"),
            show="tree headings",
        )
        comparison_tree.heading("#0", text="Strategy")
        comparison_tree.heading("TotalReturn", text="Total Return %")
        comparison_tree.heading("Sharpe", text="Sharpe Ratio")
        comparison_tree.heading("MaxDD", text="Max DD %")
        comparison_tree.heading("Trades", text="Trades")
        comparison_tree.pack(fill=tk.BOTH, expand=True)

        def run_comparison():
            selected_strategies = [
                name for name, var in strategy_vars.items() if var.get()
            ]
            if not selected_strategies:
                messagebox.showwarning("Warning", "Please select at least one strategy")
                return

            comparison_tree.delete(*comparison_tree.get_children())

            for strategy_name in selected_strategies:
                try:
                    strategy = build_strategy(strategy_name)
                    trades_df, summary, equity_df = backtest_portfolio(
                        self.data_cache, strategy
                    )

                    comparison_tree.insert(
                        "",
                        "end",
                        text=strategy_name,
                        values=(
                            f"{summary.get('TotalReturnPct', 0):.2f}",
                            (
                                f"{summary.get('Sharpe', 0):.3f}"
                                if summary.get("Sharpe") != "NA"
                                else "NA"
                            ),
                            f"{summary.get('MaxDDPct', 0):.2f}",
                            summary.get("Trades", 0),
                        ),
                    )
                except Exception as e:
                    self.log_message(f"Error comparing {strategy_name}: {e}", "ERROR")

        ttk.Button(
            comparison_window, text="Run Comparison", command=run_comparison
        ).pack(pady=10)

    def run_optimization(self):
        """Quick access to optimization tab"""
        # Switch to optimization tab
        self.notebook.select(2)  # Optimization tab index

    def show_help(self):
        """Show help dialog"""
        help_text = """
IBKR Trading Strategy Manager Pro - User Guide

🔗 Connection Management:
• Use Connect/Disconnect buttons in toolbar or Connection menu
• Test connection to verify settings before trading
• Connection status shown in toolbar (green = connected, red = disconnected)

📥 Data Management:
• Download historical data for all symbols using Data tab
• Configure date range and timeframes in left panel
• Data is cached locally for faster access
• Update existing data with delta downloads

📊 Backtesting:
• Run full backtests with historical data
• Quick backtest for rapid strategy validation
• Generate HTML and JSON reports automatically
• View detailed analytics and charts

🎯 Optimization:
• Optimize strategy parameters for best performance
• Define parameter ranges using comma-separated values
• Choose optimization metric (return, Sharpe, etc.)
• Export optimization results to Excel/CSV

🎯 Paper Trading:
• Place orders on paper trading account
• Roll orders after market close automatically
• Preview orders before placement
• Monitor current signals and positions

📈 Analytics:
• Interactive charts with zoom and pan
• Equity curve analysis
• Trade distribution and performance metrics
• Detailed drawdown analysis

Keyboard Shortcuts:
• Ctrl+S: Save configuration
• Ctrl+R/F5: Refresh all data
• Ctrl+T: Place trading orders
• Ctrl+B: Run backtest
• Ctrl+Q: Quit application

Tips:
• Always test connection before trading
• Download data before running backtests
• Use optimization to find best parameters
• Check logs tab for detailed information
• Save configurations for different strategies
        """

        help_window = tk.Toplevel(self.root)
        help_window.title("User Guide")
        help_window.geometry("700x600")

        text_widget = scrolledtext.ScrolledText(help_window, wrap=tk.WORD)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text_widget.insert(tk.END, help_text)
        text_widget.config(state=tk.DISABLED)

    def show_shortcuts(self):
        """Show keyboard shortcuts dialog"""
        shortcuts_text = """
Keyboard Shortcuts:

Connection:
• Ctrl+Shift+C - Connect to IB
• Ctrl+Shift+D - Disconnect from IB

Data Management:
• Ctrl+D - Download data
• Ctrl+U - Update data
• Ctrl+Shift+R - Clear cache

Trading:
• Ctrl+T - Place orders
• Ctrl+P - Preview orders
• Ctrl+O - Roll orders
• Ctrl+Shift+S - Show current signals

Analysis:
• Ctrl+B - Run backtest
• Ctrl+Shift+B - Quick backtest
• Ctrl+Shift+O - Run optimization
• Ctrl+E - Export results

View:
• Ctrl+1 - Data tab
• Ctrl+2 - Backtest tab
• Ctrl+3 - Optimization tab
• Ctrl+4 - Trading tab
• Ctrl+5 - Portfolio tab
• Ctrl+6 - Analytics tab
• Ctrl+7 - Logs tab

General:
• Ctrl+S - Save configuration
• Ctrl+R/F5 - Refresh all
• Ctrl+Q - Quit application
• F1 - Show help
        """

        shortcuts_window = tk.Toplevel(self.root)
        shortcuts_window.title("Keyboard Shortcuts")
        shortcuts_window.geometry("500x400")

        text_widget = scrolledtext.ScrolledText(shortcuts_window, wrap=tk.WORD)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text_widget.insert(tk.END, shortcuts_text)
        text_widget.config(state=tk.DISABLED)

    def debug_strategy_issues(self):
        """Debug strategy registration and parameter issues"""
        debug_window = tk.Toplevel(self.root)
        debug_window.title("Strategy Debug Information")
        debug_window.geometry("800x600")

        # Create scrollable text widget
        debug_text = scrolledtext.ScrolledText(debug_window, height=30, width=80)
        debug_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        debug_info = []
        debug_info.append("=== STRATEGY DEBUG INFORMATION ===\n")

        # Show registered strategies
        debug_info.append("Registered Strategies:")
        for strategy_name in STRATEGY_REGISTRY.keys():
            debug_info.append(f"  ✓ {strategy_name}")
        debug_info.append("")

        # Show strategy parameter keys
        debug_info.append("Strategy Parameter Keys:")
        for strategy_name, params in STRATEGY_PARAM_KEYS.items():
            debug_info.append(f"  {strategy_name}:")
            for param in sorted(params):
                debug_info.append(f"    - {param}")
        debug_info.append("")

        # Test strategy instantiation
        debug_info.append("Strategy Instantiation Tests:")
        current_strategy = self.strategy_var.get()

        if current_strategy in STRATEGY_REGISTRY:
            try:
                strategy_params = self.get_strategy_params()
                debug_info.append(f"  Testing {current_strategy} with params: {strategy_params}")

                strategy = build_strategy(current_strategy, **strategy_params)
                debug_info.append(f"  ✓ Successfully created {current_strategy}")
                debug_info.append(f"    - Strategy name: {strategy.name}")
                debug_info.append(f"    - Max positions: {strategy.max_positions}")
                debug_info.append(f"    - Max exposure: {strategy.max_exposure_pct}%")
                debug_info.append(f"    - Warmup bars: {strategy.warmup_bars}")

                # Test required methods
                required_methods = ['prepare', 'is_eligible', 'next_entry_spec',
                                  'dollars_per_trade', 'shares_for_entry']
                for method in required_methods:
                    if hasattr(strategy, method):
                        debug_info.append(f"    ✓ Has method: {method}")
                    else:
                        debug_info.append(f"    ✗ Missing method: {method}")

            except Exception as e:
                debug_info.append(f"  ✗ Failed to create {current_strategy}: {str(e)}")
                debug_info.append(f"    Error type: {type(e).__name__}")
                debug_info.append(f"    Full traceback:")
                import traceback
                debug_info.append(f"    {traceback.format_exc()}")
        else:
            debug_info.append(f"  ✗ Strategy '{current_strategy}' not found in registry")

        debug_info.append("")
        debug_info.append("=== RECOMMENDATIONS ===")
        debug_info.append("If your strategy is missing:")
        debug_info.append("1. Ensure your strategy file is in the strategies/ folder")
        debug_info.append("2. Make sure it's imported in the main file")
        debug_info.append("3. Verify it's added to STRATEGY_REGISTRY")
        debug_info.append("4. Check STRATEGY_PARAM_KEYS includes your strategy")
        debug_info.append("5. Ensure your strategy inherits from BaseStrategy")
        debug_info.append("6. Implement all required abstract methods")

        # Display debug information
        debug_text.insert(tk.END, '\n'.join(debug_info))
        debug_text.config(state=tk.DISABLED)

        # Add close button
        ttk.Button(debug_window, text="Close", command=debug_window.destroy).pack(pady=10)

    def show_about(self):
        """Show about dialog"""
        about_text = """
IBKR Trading Strategy Manager Pro
Version 2.1

🚀 Advanced Features:
✓ Real-time IB connection management
✓ Historical data download with caching
✓ Strategy parameter optimization
✓ Comprehensive backtesting engine
✓ Interactive analytics and charting
✓ Paper trading automation
✓ Multi-timeframe support
✓ Portfolio monitoring
✓ Export capabilities

🛠️ Built with:
• Python 3.x
• tkinter (GUI framework)
• matplotlib (charting)
• pandas (data analysis)
• Interactive Brokers API

📈 Supported Strategies:
• ADX Squeeze Breakout
• RSI Breakout
• Extensible framework for custom strategies

💡 Professional Features:
• Configuration persistence
• Comprehensive logging
• Error handling and recovery
• Multi-threading for responsiveness
• Keyboard shortcuts
• Context-sensitive help

Created for serious traders who demand
professional-grade tools and reliability.
        """

        messagebox.showinfo("About Trading Manager Pro", about_text)

    # Utility and helper methods
    def refresh_all(self):
        """Refresh all data and views"""
        self.update_status("Refreshing all data...")
        self.log_message("Refreshing all data")

        # Refresh current signals if trading tab is active
        current_tab = self.notebook.tab(self.notebook.select(), "text")
        if "Trading" in current_tab:
            self.show_current_signals()

        # Refresh results if available
        self.refresh_results()

        # Update symbol count
        self.update_symbol_count()

        # Refresh data status
        self.refresh_data_status()

        self.update_status("Refresh complete")

    def refresh_results(self):
        """Refresh results display"""
        summary_path = os.path.join(self.output_dir_var.get(), "summary.json")
        if os.path.exists(summary_path):
            try:
                with open(summary_path, "r") as f:
                    summary = json.load(f)
                self.display_summary(summary)
                self.backtest_results = {"summary": summary}
            except Exception as e:
                self.log_message(f"Failed to load summary: {e}", "ERROR")
        else:
            self.quick_results_tree.delete(*self.quick_results_tree.get_children())

    def refresh_data_status(self):
        """Refresh data status display"""
        self.data_status_tree.delete(*self.data_status_tree.get_children())

        for symbol, df in self.data_cache.items():
            status = "✓ Cached"
            bars = len(df)
            date_range = f"{df.index.min().date()} to {df.index.max().date()}"

            self.data_status_tree.insert(
                "", "end", text=symbol, values=(status, "In Memory", bars, date_range)
            )

    def refresh_logs(self):
        """Refresh logs display"""
        self.log_message("Logs refreshed")

    def clear_logs(self):
        """Clear logs display"""
        if messagebox.askyesno("Confirm", "Clear all logs?"):
            self.logs_text.config(state=tk.NORMAL)
            self.logs_text.delete(1.0, tk.END)
            self.logs_text.config(state=tk.DISABLED)
            self.log_message("Logs cleared")

    def save_logs(self):
        """Save logs to file"""
        filename = filedialog.asksaveasfilename(
            title="Save Logs",
            defaultextension=".log",
            filetypes=[
                ("Log files", "*.log"),
                ("Text files", "*.txt"),
                ("All files", "*.*"),
            ],
        )
        if filename:
            try:
                with open(filename, "w") as f:
                    f.write(self.logs_text.get(1.0, tk.END))
                messagebox.showinfo("Success", f"Logs saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save logs: {e}")

    def display_summary(self, summary):
        """Display summary in results tree"""
        self.quick_results_tree.delete(*self.quick_results_tree.get_children())

        # Color-code different types of metrics
        for key, value in summary.items():
            item_id = self.quick_results_tree.insert(
                "", "end", text=key, values=(str(value),)
            )

            # Color coding based on metric type and value
            try:
                if "Return" in key or "CAGR" in key:
                    if isinstance(value, (int, float)) and value > 0:
                        # Positive returns in green
                        pass
                elif "MaxDD" in key:
                    if isinstance(value, (int, float)) and value > 10:
                        # High drawdown in red
                        pass
            except:
                pass

    def update_signals_display(self, signals):
        """Update signals display"""
        self.signals_tree.delete(*self.signals_tree.get_children())

        for signal in signals:
            eligible_text = "✓" if signal["eligible"] else "✗"
            style = "success" if signal["eligible"] else "warning"

            item_id = self.signals_tree.insert(
                "",
                "end",
                text=signal["symbol"],
                values=(
                    (
                        f"${signal['entry_price']:.2f}"
                        if signal["entry_price"] > 0
                        else "N/A"
                    ),
                    f"${signal['stop_loss']:.2f}" if signal["stop_loss"] > 0 else "N/A",
                    signal["quantity"] if signal["quantity"] > 0 else "N/A",
                    f"${signal['risk']:.2f}" if signal["risk"] > 0 else "N/A",
                    eligible_text,
                ),
            )

    def update_status(self, message):
        """Update status bar"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_bar.config(text=f"{timestamp} - {message}")

    def auto_refresh_timer(self):
        """Auto-refresh timer for real-time updates"""
        # Refresh every 30 seconds when market is open
        hour = datetime.now().hour
        if 9 <= hour <= 16:  # Market hours
            refresh_interval = 30000  # 30 seconds
        else:
            refresh_interval = 300000  # 5 minutes

        self.root.after(refresh_interval, self.auto_refresh_timer)

        # Periodic data refresh if connected
        if self.ib_connected and hour >= 9 and hour <= 16:
            current_tab = self.notebook.tab(self.notebook.select(), "text")
            if "Trading" in current_tab:
                # Auto-refresh signals during market hours
                try:
                    self.show_current_signals()
                except:
                    pass

    # Thread management
    def run_in_thread(self, target, *args, **kwargs):
        """Run a function in a separate thread and handle output"""

        def wrapper():
            try:
                target(*args, **kwargs)
            except Exception as e:
                self.message_queue.put(
                    ("error", f"Error: {str(e)}\n{traceback.format_exc()}")
                )

        thread = threading.Thread(target=wrapper, daemon=True)
        thread.start()

    def process_messages(self):
        """Process messages from worker threads"""
        try:
            while True:
                msg_type, msg_data = self.message_queue.get_nowait()

                if msg_type == "output":
                    current_tab = self.notebook.tab(self.notebook.select(), "text")
                    if "Backtest" in current_tab:
                        self.backtest_output.insert(tk.END, msg_data)
                        self.backtest_output.see(tk.END)
                    else:
                        self.log_message(msg_data.strip())

                elif msg_type == "trading_output":
                    self.trading_output.insert(tk.END, msg_data)
                    self.trading_output.see(tk.END)

                elif msg_type == "error":
                    messagebox.showerror("Error", msg_data)
                    self.update_status("Error occurred")
                    self.log_message(msg_data, "ERROR")

                elif msg_type == "success":
                    messagebox.showinfo("Success", msg_data)
                    self.update_status("Operation completed successfully")
                    self.log_message(msg_data, "SUCCESS")

                elif msg_type == "info":
                    self.update_status(msg_data)
                    self.log_message(msg_data)

                elif msg_type == "progress":
                    self.progress_label.config(text=msg_data)
                    self.update_status(msg_data)

                elif msg_type == "data_progress":
                    self.data_progress_label.config(text=msg_data)
                    self.update_status(msg_data)

                elif msg_type == "data_status_update":
                    data = msg_data
                    self.data_status_tree.insert(
                        "",
                        "end",
                        text=data["symbol"],
                        values=(
                            data["status"],
                            data["last_updated"],
                            data["bars"],
                            data["date_range"],
                        ),
                    )

                elif msg_type == "data_download_complete":
                    self.data_progress.config(value=0)
                    self.data_progress_label.config(text="Complete")
                    self.update_status(msg_data)
                    self.log_message(msg_data, "SUCCESS")
                    messagebox.showinfo("Success", msg_data)

                elif msg_type == "backtest_complete":
                    self.backtest_results = msg_data
                    self.display_summary(msg_data["summary"])
                    self.update_status("Backtest completed successfully")
                    self.log_message("Backtest completed successfully", "SUCCESS")

                elif msg_type == "optimization_results":
                    self.display_optimization_results(msg_data)
                    self.opt_progress.config(value=100)
                    self.opt_progress_label.config(text="Optimization Complete")

                elif msg_type == "signals_update":
                    self.update_signals_display(msg_data)

                elif msg_type == "connection_success":
                    self.connection_status.config(
                        text="● Connected", style="Connected.TLabel"
                    )
                    self.ib_connected = True
                    self.update_status("IB connection successful")
                    self.log_message("IB connection successful", "SUCCESS")

                elif msg_type == "connection_error":
                    self.connection_status.config(
                        text="● Connection Failed", style="Disconnected.TLabel"
                    )
                    self.ib_connected = False
                    self.update_status("IB connection failed")
                    self.log_message(f"IB connection failed: {msg_data}", "ERROR")
                    messagebox.showerror("Connection Error", msg_data)

                elif msg_type == "validation_complete":
                    valid = msg_data["valid"]
                    invalid = msg_data["invalid"]
                    message = f"Validation complete: {len(valid)} valid, {len(invalid)} invalid symbols"
                    if invalid:
                        message += (
                            f"\nInvalid: {', '.join(invalid[:5])}"  # Show first 5
                        )
                        if len(invalid) > 5:
                            message += f" and {len(invalid)-5} more..."
                    messagebox.showinfo("Validation Results", message)
                    self.log_message(message)

                elif msg_type == "auto_open":
                    self.view_html_report()

                elif msg_type == "done":
                    if hasattr(self, "backtest_progress"):
                        self.backtest_progress.stop()
                    if hasattr(self, "opt_progress"):
                        self.opt_progress.config(value=0)
                    self.progress_label.config(text="Ready")
                    self.opt_progress_label.config(text="Ready")
                    self.update_status("Ready")

                elif msg_type == "wf_progress":
                    self.wf_progress_label.config(text=msg_data)
                    self.update_status(msg_data)

                elif msg_type == "wf_info":
                    self.update_status(msg_data)
                    self.log_message(msg_data)

                elif msg_type == "wf_warning":
                    self.log_message(msg_data, "WARNING")

                elif msg_type == "wf_error":
                    messagebox.showerror("Walk-Forward Error", msg_data)
                    self.update_status("Walk-forward analysis failed")
                    self.log_message(msg_data, "ERROR")

                elif msg_type == "wf_complete":
                    self.wf_progress.config(value=100)
                    self.wf_progress_label.config(text="Walk-forward analysis complete")
                    self.update_status("Walk-forward analysis completed")
                    messagebox.showinfo("Success", msg_data)

                elif msg_type == "wf_results":
                    aggregate_stats, window_results = msg_data
                    self.display_wf_results(aggregate_stats, window_results)

        except queue.Empty:
            pass

        # Schedule next check
        self.root.after(100, self.process_messages)

    def on_closing(self):
        """Handle application closing"""
        if messagebox.askokcancel(
            "Quit", "Do you want to quit the Trading Strategy Manager?"
        ):
            try:
                # Disconnect from IB if connected
                if self.ib_connected and self.ib_app:
                    self.ib_app.disconnect()

                # Save configuration before closing
                self.save_config_manually()
            except:
                pass
            finally:
                self.root.destroy()


def main():
    """Main application entry point"""
    root = tk.Tk()

    # Set application icon (if available)
    try:
        # You can add an icon file here if you have one
        # root.iconbitmap('trading_icon.ico')
        pass
    except:
        pass

    # Configure style with better defaults
    style = ttk.Style()

    # Try to use a modern theme
    available_themes = style.theme_names()
    preferred_themes = ["vista", "winnative", "clam", "alt"]

    for theme in preferred_themes:
        if theme in available_themes:
            style.theme_use(theme)
            break

    # Create and run application
    app = TradingApp(root)

    # Handle window closing
    root.protocol("WM_DELETE_WINDOW", app.on_closing)

    # Center window on screen
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")

    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    except Exception as e:
        print(f"Application error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
