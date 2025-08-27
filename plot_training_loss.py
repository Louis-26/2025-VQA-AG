#!/usr/bin/env python3
"""
Standalone script to plot training loss from existing LoRA training outputs.

This script can be used to plot loss curves from previously completed training runs,
even if they weren't originally trained with the --plot_loss flag.
"""
import argparse
import os
import json
import glob


def plot_training_loss(output_dir: str, num_epochs: int = None):
    """
    Plot training loss vs epoch from trainer state and checkpoint data.
    
    Args:
        output_dir: Directory containing training outputs and checkpoints
        num_epochs: Number of training epochs (auto-detected if None)
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Error: matplotlib not available. Install with: pip install matplotlib")
        return False
    
    # Collect loss data from trainer state and checkpoints
    loss_data = []
    epoch_data = []
    
    # Read main trainer state
    trainer_state_path = os.path.join(output_dir, "trainer_state.json")
    if os.path.exists(trainer_state_path):
        with open(trainer_state_path, 'r') as f:
            trainer_state = json.load(f)
        
        # Auto-detect num_epochs if not provided
        if num_epochs is None:
            num_epochs = int(trainer_state.get("num_train_epochs", 5))
            
        # Extract from log history
        for entry in trainer_state.get("log_history", []):
            if "loss" in entry and "epoch" in entry:
                loss_data.append(entry["loss"])
                epoch_data.append(entry["epoch"])
                
        print(f"Found {len(loss_data)} data points in main trainer state")
    else:
        print(f"Warning: trainer_state.json not found in {output_dir}")
        if num_epochs is None:
            num_epochs = 5  # Default
    
    # Read checkpoint trainer states for additional data points
    checkpoint_dirs = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    checkpoint_data_points = 0
    
    for checkpoint_dir in sorted(checkpoint_dirs):
        checkpoint_state_path = os.path.join(checkpoint_dir, "trainer_state.json")
        if os.path.exists(checkpoint_state_path):
            with open(checkpoint_state_path, 'r') as f:
                checkpoint_state = json.load(f)
                
            # Get the final epoch for this checkpoint
            final_epoch = checkpoint_state.get("epoch")
            if final_epoch is not None:
                # Find the last loss value from this checkpoint's history
                log_history = checkpoint_state.get("log_history", [])
                if log_history:
                    for entry in reversed(log_history):
                        if "loss" in entry:
                            # Only add if we don't already have this epoch
                            if final_epoch not in epoch_data:
                                loss_data.append(entry["loss"])
                                epoch_data.append(final_epoch)
                                checkpoint_data_points += 1
                            break
    
    print(f"Found {checkpoint_data_points} additional data points from {len(checkpoint_dirs)} checkpoints")
    
    if not loss_data:
        print("Error: No loss data found for plotting")
        return False
    
    # Sort by epoch
    combined = list(zip(epoch_data, loss_data))
    combined.sort(key=lambda x: x[0])
    epoch_data, loss_data = zip(*combined)
    
    print(f"Plotting {len(loss_data)} total data points")
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.plot(epoch_data, loss_data, 'b-o', linewidth=2, markersize=8, label='Training Loss')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('LoRA Training Loss vs Epoch', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Set epoch range
    plt.xlim(0, num_epochs)
    
    # Add loss values as text annotations
    for epoch, loss in zip(epoch_data, loss_data):
        plt.annotate(f'{loss:.4f}', 
                    (epoch, loss), 
                    textcoords="offset points", 
                    xytext=(0,12), 
                    ha='center',
                    fontsize=10,
                    alpha=0.8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    
    # Save the plot
    plot_path = os.path.join(output_dir, "training_loss_plot.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Training loss plot saved to: {plot_path}")
    
    # Also save loss data as CSV for further analysis
    csv_path = os.path.join(output_dir, "training_loss_data.csv")
    with open(csv_path, 'w') as f:
        f.write("epoch,loss\n")
        for epoch, loss in zip(epoch_data, loss_data):
            f.write(f"{epoch},{loss}\n")
    print(f"Training loss data saved to: {csv_path}")
    
    # Print summary statistics
    if len(loss_data) > 1:
        initial_loss = loss_data[0]
        final_loss = loss_data[-1]
        min_loss = min(loss_data)
        max_loss = max(loss_data)
        
        print(f"\n{'='*50}")
        print(f"Training Loss Summary:")
        print(f"{'='*50}")
        print(f"  Initial loss:    {initial_loss:.4f}")
        print(f"  Final loss:      {final_loss:.4f}")
        print(f"  Best loss:       {min_loss:.4f}")
        print(f"  Worst loss:      {max_loss:.4f}")
        print(f"  Total improvement: {((initial_loss - final_loss) / initial_loss * 100):.2f}%")
        print(f"  Data points:     {len(loss_data)}")
        print(f"  Epochs:          {num_epochs}")
        print(f"{'='*50}")
        
        # Calculate epoch-by-epoch improvements
        print(f"\nEpoch-by-Epoch Progress:")
        for i, (epoch, loss) in enumerate(zip(epoch_data, loss_data)):
            if i == 0:
                print(f"  Epoch {epoch:.1f}: {loss:.4f} (initial)")
            else:
                prev_loss = loss_data[i-1]
                improvement = ((prev_loss - loss) / prev_loss * 100)
                if improvement > 0:
                    print(f"  Epoch {epoch:.1f}: {loss:.4f} (↓{improvement:.1f}%)")
                else:
                    print(f"  Epoch {epoch:.1f}: {loss:.4f} (↑{abs(improvement):.1f}%)")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Plot training loss from LoRA training outputs")
    parser.add_argument("output_dir", help="Path to training output directory")
    parser.add_argument("--epochs", type=int, help="Number of epochs (auto-detected if not provided)")
    parser.add_argument("--show", action="store_true", help="Show plot in addition to saving")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        print(f"Error: Output directory does not exist: {args.output_dir}")
        return 1
    
    print(f"Plotting training loss from: {args.output_dir}")
    
    success = plot_training_loss(args.output_dir, args.epochs)
    
    if success and args.show:
        try:
            import matplotlib.pyplot as plt
            plt.show()
        except ImportError:
            print("Warning: matplotlib not available for displaying plot")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())

