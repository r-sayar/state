from datetime import datetime
import json
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from ..._cli._tx._predict import run_tx_predict, add_arguments_predict
from cell_eval import score_agg_metrics
import argparse
import logging
import os
import tempfile
from pathlib import Path
import pandas as pd
from typing import Dict, List, Optional, Union

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import Callback

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)

class CellEvalCallback(Callback):
    """
    Lightning callback for cell-eval evaluation during training
    """
    
    def __init__(
        self,
        eval_every_n_steps: int = 500,  # Evaluate every 500 steps (equivalent to ~5 epochs at 100 steps/epoch)
        plot_every_n_evals: int = 1,  # Plot after every 5. evaluation
        # pred_data_path: Optional[str] = None,
        # real_data_path: Optional[str] = None,
        control_pert: Optional[str] = None,
        pert_col: Optional[str] = None,
        eval_metrics: List[str] = None,
        output_dir: Optional[str] = None,
        save_predictions: bool = True,
        log_to_wandb: bool = True,
        verbose: bool = True,
        temp_checkpoint_name: str = "temp_eval_checkpoint.ckpt",
        profile: str = "minimal",  # For faster evaluation        
    ):
        super().__init__()
        
        self.eval_every_n_steps = eval_every_n_steps
        self.plot_every_n_evals = plot_every_n_evals
        #print(f"pred: {pred_data_path}")
        #self.pred_data_path = pred_data_path
        #print(f"real: {real_data_path}")
        #self.real_data_path = real_data_path
        self.control_pert = control_pert
        self.pert_col = pert_col
        self.eval_metrics = eval_metrics or ['overlap_at_N', 'mae', 'discrimination_score_l1']
        self.output_dir = Path(output_dir) if output_dir else None
        self.save_predictions = save_predictions
        self.log_to_wandb = log_to_wandb
        self.verbose = verbose
        self.temp_checkpoint_name = temp_checkpoint_name
        self.profile = profile
        self.agg_baseline = None
        
        # Evaluation history
        self.results_history = []  # list of all results DataFrames
        self.agg_results_history = []  # list of all agg_results DataFrames
        self.eval_metadata = []  # Metadata of all evaluation
        self.eval_counter = 0
        
        # Paths for persistent storage
        self.results_save_path = os.path.join(output_dir, "eval_results_history.pkl")
        self.agg_results_save_path = os.path.join(output_dir, "eval_agg_results_history.pkl")
        self.metadata_save_path = os.path.join(output_dir, "eval_metadata.json")
        self.plots_dir = os.path.join(output_dir, "evaluation_plots")
        
        # Create plots directory
        os.makedirs(self.plots_dir, exist_ok=True)

        # Cell-eval Setup
        self._setup_cell_eval()
        
    def _setup_cell_eval(self):
        """Initializes cell-eval integration"""
        try:
            # Dynamic import of cell-eval
            from cell_eval import MetricsEvaluator
            #from cell_eval.data import build_random_anndata, downsample_cells

            self.cell_eval_available = True
            
             
        
            #if self.pred_data_path and self.real_data_path:
            #    self.eval_enabled = True
            #    logger.info(f"Cell-eval callback enabled - Evaluation every{self.eval_every_n_steps} steps")
            #else:
            #    self.eval_enabled = False
            #    logger.warning("Cell callback: pred_data_path or real_data_path not set")
                
        except ImportError:
            self.cell_eval_available = False
            self.eval_enabled = False
            logger.warning("cell-eval not available - callback disabled")
    
    def _collect_evaluation_results(self, trainer, results, agg_results):
        """Collects and stores evaluation results"""
        
        # Metadata for this evaluation
        eval_metadata = {
            'eval_step': self.eval_counter,
            'global_step': trainer.global_step,
            'epoch': trainer.current_epoch,
            'timestamp': datetime.now().isoformat(),
            'learning_rate': trainer.optimizers[0].param_groups[0]['lr'] if trainer.optimizers else None
        }
        # Add data to history
        if results is not None and not results.is_empty():
            # Extend DataFrame with step information
            #print("Buchi results:", type(results))
            results_with_step = results.to_pandas() #pd.DataFrame(results, columns=list(results.name))
            #print("Buchi results_with_step:", results_with_step)
            results_with_step['eval_step'] = self.eval_counter
            results_with_step['global_step'] = trainer.global_step
            results_with_step['epoch'] = trainer.current_epoch
            self.results_history.append(results_with_step)
        if agg_results is not None and not agg_results.is_empty():
            # Extend DataFrame with step information
            #print("Buchi agg_results:", type(agg_results))
            #agg_results_with_step = pd.DataFrame(agg_results, columns=list(agg_results[0].keys()))
            agg_results_with_step = agg_results.to_pandas() #pd.DataFrame(agg_results, columns=agg_results.name)
            #print("Buchi agg_results_with_step:", agg_results_with_step)
            agg_results_with_step['eval_step'] = self.eval_counter
            agg_results_with_step['global_step'] = trainer.global_step
            agg_results_with_step['epoch'] = trainer.current_epoch
            self.agg_results_history.append(agg_results_with_step)
        
        self.eval_metadata.append(eval_metadata)
        self.eval_counter += 1
        
        # Store data persistently
        self._save_data()
        if self.verbose:
            logger.info(f"Collected evaluation results for step {trainer.global_step} "
                    f"(eval #{self.eval_counter})")

    def _save_data(self):
        """Stores all collected data persistently"""
        try:
            # Save results history
            if self.results_history:
                with open(self.results_save_path, 'wb') as f:
                    pickle.dump(self.results_history, f)
            
            # Save Agg results
            if self.agg_results_history:
                with open(self.agg_results_save_path, 'wb') as f:
                    pickle.dump(self.agg_results_history, f)
            
            # Save metadata
            with open(self.metadata_save_path, 'w') as f:
                json.dump(self.eval_metadata, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Could not save evaluation history: {e}")

    def _load_existing_data(self):
        """Loads existing data at callback start"""
        try:
            if os.path.exists(self.results_save_path):
                with open(self.results_save_path, 'rb') as f:
                    self.results_history = pickle.load(f)
            
            if os.path.exists(self.agg_results_save_path):
                with open(self.agg_results_save_path, 'rb') as f:
                    self.agg_results_history = pickle.load(f)
            
            if os.path.exists(self.metadata_save_path):
                with open(self.metadata_save_path, 'r') as f:
                    self.eval_metadata = json.load(f)
                    
            self.eval_counter = len(self.eval_metadata)
            
            if self.eval_counter > 0:
                logger.info(f"Loaded {self.eval_counter} previous evaluations")
                
        except Exception as e:
            logger.warning(f"Could not load existing evaluation history: {e}")

    def _create_evaluation_plots(self, trainer):
        """Creates robust plots based on available data"""
        
        if len(self.agg_results_history) < 2:
            logger.info("Need at least 2 evaluations for meaningful plots")
            return
        
        try:
            # Creates combined DataFrames
            combined_agg_results = pd.concat(self.agg_results_history, ignore_index=True)
            
            logger.info(f"Available agg_results columns: {list(combined_agg_results.columns)}")
            
            #print(f"Buchi _create_evaluation_plots combined_agg_results: {combined_agg_results}")
            # Set plot style
            plt.style.use('default')  # Fallback falls seaborn nicht verfügbar
            
            # 1. Available aggregated metrics over time
            self._plot_aggregated_metrics_over_time(combined_agg_results, trainer.global_step)
            
            # 2. Detailed analysis of current metrics
            self._plot_current_metrics_analysis(combined_agg_results, trainer.global_step)
            
            # 3. Perturbation analysis (if results are available)
            if self.results_history:
                combined_results = pd.concat(self.results_history, ignore_index=True)
                logger.info(f"Available results columns: {list(combined_results.columns)}")
                self._plot_perturbation_metrics_robust(combined_results, trainer.global_step)
            
            logger.info(f"Successfully created evaluation plots for step {trainer.global_step}")
            
        except Exception as e:
            logger.error(f"Error creating evaluation plots: {e}")
            import traceback
            traceback.print_exc()
    def _plot_aggregated_metrics_over_time(self, combined_agg_results, current_step):
        """Plot der aggregierten Metriken über die Zeit - angepasst für verfügbare Spalten"""
        #print("Buchi _plot_aggregated_metrics_over_time")
        # Identify available metrics
        available_metrics = []
        possible_metrics = ['overlap_at_N', 'mae', 'discrimination_score_l1', 'pearson_delta', 'mse']
        
        #print(f"Buchi _plot_aggregated_metrics_over_time combined_agg_results: {combined_agg_results}")
        for metric in possible_metrics:
            if metric in combined_agg_results.columns:
                available_metrics.append(metric)
        
        if not available_metrics:
            logger.warning("No metrics available for plotting")
            return
        
        # Dynamic plot size based on available metrics
        n_metrics = len(available_metrics)
        n_cols = min(2, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 5*n_rows))
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'Available Metrics Over Training Steps (Current: {current_step})', 
                    fontsize=16, fontweight='bold')
        
        for idx, metric in enumerate(available_metrics):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            # Line plot with markers
            sns.lineplot(data=combined_agg_results, x='global_step', y=metric, 
                        ax=ax, marker='o', linewidth=2, markersize=6)
            
            ax.set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
            ax.set_xlabel('Global Training Step')
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.grid(True, alpha=0.3)
            
            # Add trend line
            if len(combined_agg_results) > 2:
                try:
                    z = np.polyfit(combined_agg_results['global_step'], 
                                combined_agg_results[metric], 1)
                    p = np.poly1d(z)
                    ax.plot(combined_agg_results['global_step'], 
                        p(combined_agg_results['global_step']), 
                        "--", alpha=0.7, color='red', label='Trend')
                    ax.legend()
                except:
                    pass  # If Polyfit fails
        
        # Hide empty subplots
        if n_metrics < n_rows * n_cols:
            for idx in range(n_metrics, n_rows * n_cols):
                row = idx // n_cols
                col = idx % n_cols
                ax = axes[row, col] if n_rows > 1 else axes[col]
                ax.set_visible(False)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.plots_dir, f'available_metrics_step_{current_step}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved metrics plot with {n_metrics} available metrics: {save_path}")

    def _plot_current_metrics_analysis(self, combined_agg_results, current_step):
        """Special analysis for overlap_at_N, mae, discrimination_score_l1"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Detailed Metrics Analysis (Step: {current_step})', 
                    fontsize=16, fontweight='bold')
        
        # 1. All three metrics normalized in one plot
        ax1 = axes[0, 0]
        
        #print(f"Buchi _plot_current_metrics_analysis combined_agg_results: {combined_agg_results}")
        # Normalize data for comparability
        metrics_to_normalize = ['overlap_at_N', 'mae', 'discrimination_score_l1']
        available_metrics = [m for m in metrics_to_normalize if m in combined_agg_results.columns]
        
        for metric in available_metrics:
            # Min-Max Normalization
            values = combined_agg_results[metric]
            normalized_values = (values - values.min()) / (values.max() - values.min()) if values.max() != values.min() else values
            
            ax1.plot(combined_agg_results['global_step'], normalized_values, 
                    marker='o', label=f'{metric} (normalized)', linewidth=2)
        
        ax1.set_title('Normalized Metrics Comparison')
        ax1.set_xlabel('Global Training Step')
        ax1.set_ylabel('Normalized Value [0-1]')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. MAE detailed analysis
        ax2 = axes[0, 1]
        if 'mae' in combined_agg_results.columns:
            mae_values = combined_agg_results['mae']
            ax2.plot(combined_agg_results['global_step'], mae_values, 
                    'b-o', linewidth=2, markersize=6)
            ax2.set_title('MAE Progression')
            ax2.set_xlabel('Global Training Step')
            ax2.set_ylabel('Mean Absolute Error')
            ax2.grid(True, alpha=0.3)
            
            # Mark best and worst MAE
            best_idx = mae_values.idxmin()
            worst_idx = mae_values.idxmax()
            ax2.scatter(combined_agg_results.loc[best_idx, 'global_step'], 
                    mae_values.iloc[best_idx], color='green', s=100, 
                    label=f'Best: {mae_values.iloc[best_idx]:.4f}', zorder=5)
            ax2.scatter(combined_agg_results.loc[worst_idx, 'global_step'], 
                    mae_values.iloc[worst_idx], color='red', s=100, 
                    label=f'Worst: {mae_values.iloc[worst_idx]:.4f}', zorder=5)
            ax2.legend()
        
        # 3. Discrimination score analysis
        ax3 = axes[1, 0]
        if 'discrimination_score_l1' in combined_agg_results.columns:
            disc_values = combined_agg_results['discrimination_score_l1']
            ax3.plot(combined_agg_results['global_step'], disc_values, 
                    'g-o', linewidth=2, markersize=6)
            ax3.set_title('Discrimination Score L1 Progression')
            ax3.set_xlabel('Global Training Step')
            ax3.set_ylabel('Discrimination Score L1')
            ax3.grid(True, alpha=0.3)
            
            # average line
            mean_score = disc_values.mean()
            ax3.axhline(y=mean_score, color='orange', linestyle='--', 
                    label=f'Mean: {mean_score:.3f}')
            ax3.legend()
        
        # 4. Overlap at N analysis
        ax4 = axes[1, 1]
        if 'overlap_at_N' in combined_agg_results.columns:
            overlap_values = combined_agg_results['overlap_at_N']
            ax4.plot(combined_agg_results['global_step'], overlap_values, 
                    'r-o', linewidth=2, markersize=6)
            ax4.set_title('Overlap at N Progression')
            ax4.set_xlabel('Global Training Step')
            ax4.set_ylabel('Overlap at N')
            ax4.grid(True, alpha=0.3)
            
            # Also show the distribution as a histogram in the inset
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            inset_ax = inset_axes(ax4, width="30%", height="30%", loc='upper right')
            inset_ax.hist(overlap_values, bins=10, alpha=0.7, color='red')
            inset_ax.set_title('Distribution', fontsize=8)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.plots_dir, f'detailed_analysis_step_{current_step}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved detailed analysis plot: {save_path}")

    def _plot_perturbation_metrics_robust(self, combined_results, current_step):
        """Robust perturbation analysis for available columns"""
        #print(f"Buchi _plot_perturbation_metrics_robust combined_results: {combined_results}")
        
        if combined_results is None or len(combined_results) == 0:
            logger.warning("No perturbation results available for plotting")
            return
        
        # Last evaluation
        latest_eval = combined_results[combined_results['eval_step'] == combined_results['eval_step'].max()]
        
        # Identify available metrics
        available_metrics = []
        possible_metrics = ['overlap_at_N', 'mae', 'discrimination_score_l1']
        
        for metric in possible_metrics:
            if metric in latest_eval.columns:
                available_metrics.append(metric)
        
        if not available_metrics:
            logger.warning("No perturbation metrics available for plotting")
            return
        
        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 8))
        if n_metrics == 1:
            axes = [axes]
        
        fig.suptitle(f'Top Perturbations Analysis (Step: {current_step})', 
                    fontsize=16, fontweight='bold')
        
        for idx, metric in enumerate(available_metrics):
            ax = axes[idx]
            
            # Top 20 Perturbations for Better Visibility
            if metric in ['discrimination_score_l1', 'overlap_at_N']:
                # Higher values are better
                top_perts = latest_eval.nlargest(20, metric)
                title_suffix = "(Higher is Better)"
            else:
                # Lower values are better (mae)
                top_perts = latest_eval.nsmallest(20, metric)
                title_suffix = "(Lower is Better)"
            
            # Horizontal bar plot
            y_pos = np.arange(len(top_perts))
            bars = ax.barh(y_pos, top_perts[metric])
            
            # Set perturbation names
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_perts['perturbation'], fontsize=8)
            ax.set_xlabel(f'{metric.replace("_", " ").title()}')
            ax.set_title(f'Top 20 Perturbations\n{metric.replace("_", " ").title()} {title_suffix}')
            ax.grid(True, alpha=0.3, axis='x')
            
            # color gradient
            colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            # Add values as text
            for i, (bar, value) in enumerate(zip(bars, top_perts[metric])):
                ax.text(bar.get_width() + bar.get_width()*0.01, bar.get_y() + bar.get_height()/2, 
                    f'{value:.3f}', ha='left', va='center', fontsize=7)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.plots_dir, f'perturbation_analysis_robust_step_{current_step}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved robust perturbation analysis: {save_path}")

    def on_train_batch_end(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule, 
        outputs, 
        batch, 
        batch_idx: int
    ) -> None:
        """
        Called after each training batch
        old
        """
        if trainer.global_step % self.eval_every_n_steps == 0 and trainer.global_step > 0:            
            logger.info(f"Running dynamic evaluation at step {trainer.global_step}")
        
            # Create temporary checkpoint
            temp_checkpoint_path = self._save_temp_checkpoint(trainer)
            
            try:
                # Call run_tx_predict directly
                #metrics = self._call_run_tx_predict(temp_checkpoint_path)
                (results, agg_results) = self._call_run_tx_predict(temp_checkpoint_path)
                
                #print("Buchi after run_tx_predict")
                # Metrics loggen
                #print(f"Buchi results: {results}")
                #print(f"Buchi agg_results: {agg_results}")
        
                #if metrics and self.log_to_wandb:
                if self.log_to_wandb:
                    print("Buchi Yeah metrics log!")
                    #self._log_metrics(trainer, results, agg_results)
                    #self._process_eval_results(trainer, results, trainer.global_step)
                            
            except Exception as e:
                logger.error(f"Error during dynamic evaluation: {e}")
            finally:
                # Cleanup
                #self._cleanup_temp_files(temp_checkpoint_path)
                print("no cleanup necessary!")

    def _save_temp_checkpoint(self, trainer):
        """Saves temporary checkpoint for run_tx_predict"""
        temp_dir = os.path.join(self.output_dir, "temp_eval")
        os.makedirs(temp_dir, exist_ok=True)
        temp_checkpoint_path = os.path.join(temp_dir, self.temp_checkpoint_name)
        
        trainer.save_checkpoint(temp_checkpoint_path)
        return temp_checkpoint_path
    def _call_run_tx_predict(self, checkpoint_path):
        """Call run_tx_predict with the correct arguments"""
        
        # Create arguments for run_tx_predict
        class MockArgs:
            def __init__(self, output_dir, checkpoint, profile="minimal", predict_only=False):
                self.output_dir = output_dir
                self.checkpoint = os.path.basename(checkpoint)  # only filename
                self.test_time_finetune = 0  # No fine-tuning during training
                self.profile = profile
                self.predict_only = predict_only
        
        # Move checkpoint to the expected directory
        expected_checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(expected_checkpoint_dir, exist_ok=True)
        expected_checkpoint_path = os.path.join(expected_checkpoint_dir, self.temp_checkpoint_name)
        
        # copy checkpoint
        import shutil
        shutil.copy2(checkpoint_path, expected_checkpoint_path)
        
        try:
            # call cell_eval baseline
            if self.agg_baseline is None:                
                logger.info("call Cell-eval baseline.")
                # create mock arguments
                args = MockArgs(
                    output_dir=self.output_dir,
                    checkpoint=expected_checkpoint_path,                
                    profile="vcc",
                    predict_only=False
                )
                res_baseline, self.agg_baseline = run_tx_predict(args)
                logger.info(f"Cell-eval baseline results: {self.agg_baseline}")
            # create mock arguments
            args = MockArgs(
                output_dir=self.output_dir,
                checkpoint=expected_checkpoint_path,
                profile="vcc",
                predict_only=False
            )
            
            # run_tx_predict aufrufen
            (results, agg_results) = run_tx_predict(args)
            #print("Buchi after run_tx_predict in _call_run_tx_predict")
            return (results, agg_results)
            
        finally:
            # Delete temporary checkpoint from checkpoints/
            if os.path.exists(expected_checkpoint_path):
                os.remove(expected_checkpoint_path)

    """def _run_evaluation(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule, 
        current_step: int
    ):
        
        Performs the cell evaluation
        
        try:
            # Modell in Evaluierungs-Modus setzen
            pl_module.eval()
            
            # Temporäres Checkpoint erstellen
            with tempfile.NamedTemporaryFile(suffix='.ckpt', delete=False) as tmp_file:
                temp_checkpoint_path = tmp_file.name
                
            # Checkpoint speichern
            trainer.save_checkpoint(temp_checkpoint_path)
            
            # Cell-eval Evaluierung durchführen
            eval_results = self._perform_cell_eval(
                checkpoint_path=temp_checkpoint_path,
                step=current_step
            )
            
            # Temporäres Checkpoint löschen
            os.unlink(temp_checkpoint_path)
            
            # Ergebnisse verarbeiten
            if eval_results:
                self._process_eval_results(trainer, eval_results, current_step)
            
            # Modell zurück in Training-Modus
            pl_module.train()
            
        except Exception as e:
            logger.error(f"Fehler bei cell-eval Evaluierung bei Step {current_step}: {e}")
    
    def _perform_cell_eval(
        self, 
        checkpoint_path: str, 
        step: int
    ) -> Optional[Dict]:
        
        Performs the actual cell evaluation
        
        try:
            from cell_eval import MetricsEvaluator
            
            # Initialize Cell-eval Evaluator
            evaluator = MetricsEvaluator(
                adata_pred=self.pred_data_path,
                adata_real=self.real_data_path,
                control_pert=self.control_pert, #"control",
                pert_col=self.pert_col, #"non-targeting",
                num_threads=64,
            )
            (results, agg_results) = evaluator.compute()
            
            return results
            
        except Exception as e:
            logger.error(f"Cell-eval Evaluierung fehlgeschlagen: {e}")
            return None
    """
    """def _process_eval_results(
        self, 
        trainer: pl.Trainer, 
        eval_results: Dict, 
        current_step: int
    ):
        Verarbeitet und loggt die Evaluierungs-Ergebnisse
        # Ergebnisse zur Historie hinzufügen
        self.eval_history.append({
            "step": current_step,
            "results": eval_results
        })
        
        # Console Logging
        if self.verbose:
            logger.info(f"Cell-eval Ergebnisse für Step {current_step}:")
            for metric, value in eval_results.items():
                logger.info(f"  {metric}: {value:.4f}")
        
        # Lightning Logging
        log_dict = {f"eval/{metric}": value for metric, value in eval_results.items()}
        log_dict["eval/step"] = current_step
        trainer.log_dict(log_dict, step=current_step)
        
        # Wandb Logging
        if self.log_to_wandb and WANDB_AVAILABLE and wandb.run:
            wandb_log = {f"cell_eval/{metric}": value for metric, value in eval_results.items()}
            wandb_log["step"] = current_step
            wandb.log(wandb_log, step=current_step)
        
        # Plots erstellen und speichern
        if len(self.eval_history) > 1:
            self._create_evaluation_plots(current_step)
    
    def _create_evaluation_plots(self, current_step: int):
        
        Erstellt Plots für die Evaluierungs-Metriken über die Zeit
        
        try:
            import matplotlib.pyplot as plt
            
            if not self.output_dir:
                return
                
            steps = [h["step"] for h in self.eval_history]
            
            # Plot für jede Metrik
            n_metrics = len(self.eval_metrics)
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            axes = axes.flatten() if n_metrics > 1 else [axes]
            
            for i, metric in enumerate(self.eval_metrics[:4]):  # Max 4 Plots
                if i < len(axes):
                    values = [h["results"].get(metric, 0) for h in self.eval_history]
                    axes[i].plot(steps, values, marker='o', linewidth=2)
                    axes[i].set_title(f'{metric.upper()} über Training Steps')
                    axes[i].set_xlabel('Training Step')
                    axes[i].set_ylabel(metric.upper())
                    axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Plot speichern
            plot_dir = self.output_dir / "cell_eval_plots"
            plot_dir.mkdir(exist_ok=True)
            plot_path = plot_dir / f"metrics_step_{current_step}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Zu Wandb loggen
            if self.log_to_wandb and WANDB_AVAILABLE and wandb.run:
                wandb.log({
                    f"cell_eval_plots/step_{current_step}": wandb.Image(str(plot_path))
                }, step=current_step)
                
        except Exception as e:
            logger.error(f"Fehler beim Erstellen der Evaluierungs-Plots: {e}")
    """
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.eval_every_n_steps == 0 and trainer.global_step > 0:
            logger.info(f"Running dynamic evaluation at step {trainer.global_step}")
            
            temp_checkpoint_path = self._save_temp_checkpoint(trainer)
            
            try:
                # Call run_tx_predict and collect results
                results, agg_results = self._call_run_tx_predict(temp_checkpoint_path)
                
                # Collect and store results
                self._collect_evaluation_results(trainer, results, agg_results)
                
                # log metrics
                #if agg_results is not None and self.log_to_wandb:
                #    self._log_metrics(trainer, agg_results)
                
                # run score
                score_filename = "score_step_" + str(self.eval_counter) + ".csv"
                score_agg_metrics(
                    results_user=agg_results,
                    results_base=self.agg_baseline,                    
                    output=os.path.join(self.output_dir, score_filename)
                )
                logger.info(f"Buchi calc score_agg_metrics in step: {self.eval_counter}")
    
                # Create plots (all N evaluations)
                if self.eval_counter % self.plot_every_n_evals == 0:
                    logger.info(f"Creating evaluation plots (eval #{self.eval_counter})")
                    self._create_evaluation_plots(trainer)
                    
            except Exception as e:
                logger.error(f"Error during dynamic evaluation: {e}")
            finally:
                self._cleanup_temp_files(temp_checkpoint_path)

    def get_combined_dataframes(self):
        """Returns the combined DataFrames for external analysis"""
        combined_results = None
        combined_agg_results = None
        
        if self.results_history:
            combined_results = pd.concat(self.results_history, ignore_index=True)
        
        if self.agg_results_history:
            combined_agg_results = pd.concat(self.agg_results_history, ignore_index=True)
        
        return combined_results, combined_agg_results

    def _cleanup_temp_files(self, checkpoint_path):
        """Cleanup of temporary files"""
        try:
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
            temp_dir = os.path.dirname(checkpoint_path)
            if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                os.rmdir(temp_dir)
        except Exception as e:
            logger.warning(f"Could not cleanup temp files: {e}")

    def _log_metrics(self, trainer, metrics):
        """Logs the metrics from run_tx_predict"""
        if not metrics:
            return
            
        # Forward metrics to Lightning Logger
        for metric_name, metric_value in metrics.items():
            trainer.logger.log_metrics(
                {f"eval_{metric_name}": metric_value}, 
                step=trainer.global_step
            )
        
        if self.verbose:
            logger.info(f"Step {trainer.global_step} - Evaluation metrics: {metrics}")