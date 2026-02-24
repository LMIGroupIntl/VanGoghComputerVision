import argparse
import torch
import torch.nn.functional as F
import timm
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import json
from datetime import datetime
from typing import List, Dict
import os

class VanGoghInference:
    def __init__(self, model_paths: List[str], device=None, target_size=256, aggregation='majority'):
        """Initialize the inference system with model paths.
        
        Args:
            model_paths: List of paths to model files or directories
            device: Device to use (cuda, mps, cpu)
            target_size: Target image size for preprocessing
            aggregation: Aggregation method - 'majority', 'mean', or 'both' (default: 'majority')
        """
        self.target_size = target_size
        self.aggregation = aggregation
        
        # Validate aggregation method
        if self.aggregation not in ['majority', 'mean', 'both']:
            raise ValueError(f"Invalid aggregation method: {self.aggregation}. Must be 'majority', 'mean', or 'both'")
        
        # Setup device
        if device is None:
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")
        print(f"Aggregation method: {self.aggregation}")

        self.class_to_idx = {'authentic': 0, 'imitation': 1}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        # Process model paths and load models
        self.models = {}
        self._find_and_load_models(model_paths)

        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((self.target_size, self.target_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print(f"Successfully loaded {len(self.models)} models")
        self._print_model_summary()

    def _find_model_files(self, paths: List[str]) -> List[Path]:
        """Find all .pth model files from given paths (files or directories)."""
        model_files = []
        
        for path_str in paths:
            path = Path(path_str)
            
            if path.is_file():
                if path.suffix.lower() == '.pth':
                    model_files.append(path)
                    print(f"Added model file: {path.name}")
                else:
                    print(f"Skipping non-.pth file: {path.name}")
                    
            elif path.is_dir():
                # Find all .pth files in directory
                pth_files = list(path.glob('*.pth'))
                if pth_files:
                    model_files.extend(pth_files)
                    print(f"Found {len(pth_files)} .pth files in {path}:")
                    for pth_file in pth_files:
                        print(f"  - {pth_file.name}")
                else:
                    print(f"No .pth files found in directory: {path}")
            else:
                print(f"Warning: Path does not exist: {path}")
        
        return model_files

    def _determine_model_type(self, filename: str) -> str:
        """Determine model type from filename."""
        filename_lower = filename.lower()
        
        if 'swin' in filename_lower:
            return 'swin'
        elif 'efficientnet' in filename_lower or 'effnet' in filename_lower:
            return 'efficientnet'
        else:
            raise ValueError(f"Cannot determine model type from filename: {filename}")

    def _determine_category(self, filename: str) -> str:
        """Determine model category from filename."""
        filename_lower = filename.lower()
        
        if 'stable' in filename_lower:
            return 'stable'
        elif 'unstable' in filename_lower:
            return 'unstable'
        elif 'regularized' in filename_lower:
            return 'regularized'
        elif 'overfit' in filename_lower:
            return 'overfit'
        else:
            return 'unknown'

    def _count_parameters(self, model):
        """Count model parameters."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params, trainable_params

    def _format_param_count(self, param_count):
        """Format parameter count with units."""
        if param_count >= 1e9:
            return f"{param_count/1e9:.2f}B"
        elif param_count >= 1e6:
            return f"{param_count/1e6:.2f}M"
        elif param_count >= 1e3:
            return f"{param_count/1e3:.2f}K"
        else:
            return str(param_count)

    def _find_and_load_models(self, model_paths: List[str]):
        """Find model files and load them."""
        model_files = self._find_model_files(model_paths)
        
        if not model_files:
            raise ValueError("No valid .pth model files found")
        
        print(f"\nLoading {len(model_files)} models...")
        
        for model_file in model_files:
            try:
                model_type = self._determine_model_type(model_file.name)
                category = self._determine_category(model_file.name)
                model_name = model_file.stem
                
                print(f"Loading {model_name} ({model_type}, {category})...")
                
                # Create model architecture
                if model_type == 'swin':
                    model = timm.create_model(
                        'swin_tiny_patch4_window7_224',
                        pretrained=False,
                        num_classes=2,
                        img_size=256,
                        drop_rate=0.5,
                        drop_path_rate=0.4
                    )
                elif model_type == 'efficientnet':
                    model = timm.create_model(
                        'efficientnet_b5',
                        pretrained=False,
                        num_classes=2
                    )
                
                # Load weights
                state_dict = torch.load(model_file, map_location=self.device)
                model.load_state_dict(state_dict)
                model.to(self.device)
                model.eval()
                
                # Count parameters
                total_params, trainable_params = self._count_parameters(model)
                
                self.models[model_name] = {
                    'model': model,
                    'type': model_type,
                    'category': category,
                    'path': str(model_file),
                    'total_params': total_params,
                    'trainable_params': trainable_params
                }
                
                print(f"✓ {model_name} loaded - {self._format_param_count(total_params)} parameters")
                
            except Exception as e:
                print(f"✗ Error loading {model_file.name}: {e}")
                continue

    def _print_model_summary(self):
        """Print summary of loaded models."""
        print("\n" + "="*80)
        print("MODEL SUMMARY")
        print("="*80)
        
        total_system_params = 0
        
        for model_name, info in self.models.items():
            total_params = info['total_params']
            total_system_params += total_params
            
            print(f"Model: {model_name}")
            print(f"  Type: {info['type']}")
            print(f"  Category: {info['category']}")
            print(f"  Parameters: {self._format_param_count(total_params)} ({total_params:,})")
            print(f"  Path: {info['path']}")
            print()
        
        print(f"TOTAL SYSTEM PARAMETERS: {self._format_param_count(total_system_params)} ({total_system_params:,})")
        print("="*80)

    def _extract_patches(self, image):
        """Extract patches from image using training logic."""
        w, h = image.size
        max_dim = max(w, h)
        
        if max_dim > 1024:
            grid_size = 4  # 4x4 patches
        elif max_dim >= 512:
            grid_size = 2  # 2x2 patches
        else:
            grid_size = 1  # 1x1 patch
        
        patches = []
        
        if grid_size == 1:
            # Single patch - center crop or resize
            min_dim = min(w, h)
            if min_dim < 256:
                resized = image.resize((256, 256), Image.Resampling.LANCZOS)
                patches.append(resized)
            else:
                left = (w - min_dim) // 2
                top = (h - min_dim) // 2
                center_crop = image.crop((left, top, left + min_dim, top + min_dim))
                patches.append(center_crop)
        else:
            # Grid-based patches
            patch_width = w // grid_size
            patch_height = h // grid_size
            
            for i in range(grid_size):
                for j in range(grid_size):
                    left = j * patch_width
                    upper = i * patch_height
                    right = (j + 1) * patch_width if (j + 1) < grid_size else w
                    bottom = (i + 1) * patch_height if (i + 1) < grid_size else h
                    
                    patch = image.crop((left, upper, right, bottom))
                    if patch.size[0] > 0 and patch.size[1] > 0:
                        patches.append(patch)
        
        return patches, grid_size

    def _predict_single_model(self, image, model_info):
        """Run inference with a single model using specified aggregation method."""
        model = model_info['model']
        patches, grid_size = self._extract_patches(image)
        
        if not patches:
            raise ValueError("No patches extracted")
        
        # Transform and predict
        all_logits = []
        patch_predictions = []
        patch_confidences = []
        
        with torch.no_grad():
            for patch in patches:
                patch_tensor = self.transform(patch).unsqueeze(0).to(self.device)
                logits = model(patch_tensor)
                probabilities = F.softmax(logits, dim=1)
                
                # Store logits for mean aggregation
                all_logits.append(logits)
                
                # Get individual patch prediction for majority voting
                predicted_class_idx = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities.cpu().numpy()[0]
                
                patch_predictions.append(predicted_class_idx)
                patch_confidences.append(confidence)
        
        results = {
            'num_patches': len(patches),
            'grid_size': grid_size,
            'model_type': model_info['type'],
            'model_category': model_info['category']
        }
        
        # Method 1: Mean Logits Aggregation
        if self.aggregation in ['mean', 'both']:
            stacked_logits = torch.cat(all_logits, dim=0)
            avg_logits = torch.mean(stacked_logits, dim=0, keepdim=True)
            mean_probabilities = F.softmax(avg_logits, dim=1)
            
            mean_predicted_idx = torch.argmax(mean_probabilities, dim=1).item()
            mean_predicted_class = self.idx_to_class[mean_predicted_idx]
            mean_confidence = float(mean_probabilities[0, mean_predicted_idx])
            
            results['mean_logits'] = {
                'predicted_class': mean_predicted_class,
                'confidence': mean_confidence,
                'probabilities': {
                    'authentic': float(mean_probabilities[0, 0]),
                    'imitation': float(mean_probabilities[0, 1])
                }
            }
        
        # Method 2: Majority Voting
        if self.aggregation in ['majority', 'both']:
            # Count votes
            authentic_votes = sum(1 for pred in patch_predictions if pred == 0)
            imitation_votes = sum(1 for pred in patch_predictions if pred == 1)
            
            # Determine final prediction based on majority
            if authentic_votes > imitation_votes:
                majority_final_class = 'authentic'
            elif imitation_votes > authentic_votes:
                majority_final_class = 'imitation'
            else:
                # Tie case - use average confidence to break tie
                avg_authentic_conf = np.mean([conf[0] for conf in patch_confidences])
                avg_imitation_conf = np.mean([conf[1] for conf in patch_confidences])
                
                if avg_authentic_conf > avg_imitation_conf:
                    majority_final_class = 'authentic'
                else:
                    majority_final_class = 'imitation'
            
            # Calculate final confidence based on the winning class
            if majority_final_class == 'authentic':
                # Average confidence of patches that predicted authentic
                authentic_confidences = [patch_confidences[i][0] for i in range(len(patch_predictions))
                                       if patch_predictions[i] == 0]
                if authentic_confidences:
                    majority_final_confidence = np.mean(authentic_confidences)
                else:
                    majority_final_confidence = np.mean([conf[0] for conf in patch_confidences])
            else:
                # Average confidence of patches that predicted imitation
                imitation_confidences = [patch_confidences[i][1] for i in range(len(patch_predictions))
                                       if patch_predictions[i] == 1]
                if imitation_confidences:
                    majority_final_confidence = np.mean(imitation_confidences)
                else:
                    majority_final_confidence = np.mean([conf[1] for conf in patch_confidences])
            
            # Calculate overall probabilities for reporting (vote fractions)
            overall_authentic_prob = authentic_votes / len(patch_predictions)
            overall_imitation_prob = imitation_votes / len(patch_predictions)
            
            results['majority_voting'] = {
                'predicted_class': majority_final_class,
                'confidence': float(majority_final_confidence),
                'probabilities': {
                    'authentic': float(overall_authentic_prob),
                    'imitation': float(overall_imitation_prob)
                },
                'vote_counts': {
                    'authentic': authentic_votes,
                    'imitation': imitation_votes,
                    'total_patches': len(patch_predictions)
                }
            }
        
        # Set primary result based on aggregation method
        if self.aggregation == 'mean':
            results['predicted_class'] = results['mean_logits']['predicted_class']
            results['confidence'] = results['mean_logits']['confidence']
            results['probabilities'] = results['mean_logits']['probabilities']
        elif self.aggregation == 'majority':
            results['predicted_class'] = results['majority_voting']['predicted_class']
            results['confidence'] = results['majority_voting']['confidence']
            results['probabilities'] = results['majority_voting']['probabilities']
        else:  # both
            # Default to majority for primary display
            results['predicted_class'] = results['majority_voting']['predicted_class']
            results['confidence'] = results['majority_voting']['confidence']
            results['probabilities'] = results['majority_voting']['probabilities']
        
        return results

    def predict_image(self, image_path):
        """Run inference on a single image."""
        print(f"Processing: {Path(image_path).name}")
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
        
        results = {
            'image_path': str(image_path),
            'image_size': image.size,
            'timestamp': datetime.now().isoformat(),
            'aggregation_method': self.aggregation,
            'models': {}
        }
        
        # Run inference with each model
        for model_name, model_info in self.models.items():
            try:
                result = self._predict_single_model(image, model_info)
                results['models'][model_name] = result
                
                # Display based on aggregation method
                if self.aggregation == 'both':
                    print(f"  {model_name:20} ({result['model_category']:8}):")
                    if 'majority_voting' in result:
                        maj = result['majority_voting']
                        print(f"    MAJORITY: {maj['predicted_class']:10} | Conf: {maj['confidence']:.3f}")
                    if 'mean_logits' in result:
                        mean = result['mean_logits']
                        print(f"    MEAN:     {mean['predicted_class']:10} | Conf: {mean['confidence']:.3f}")
                else:
                    print(f"  {model_name:20} ({result['model_category']:8}): {result['predicted_class']:10} "
                          f"| Conf: {result['confidence']:.3f}")
                
            except Exception as e:
                print(f"  {model_name:20} ERROR: {e}")
                results['models'][model_name] = {'error': str(e)}
        
        return results

    def _print_summary_table(self, results):
        """Print a comprehensive summary table of all predictions."""
        if not results:
            print("\nNo results to display!")
            return
        
        print("\n" + "="*120)
        print(f"INFERENCE SUMMARY - AGGREGATION METHOD: {self.aggregation.upper()}")
        print("="*120)
        
        # Get all model names
        model_names = list(results[0]['models'].keys()) if results and results[0]['models'] else []
        
        if not model_names:
            print("\n⚠️  WARNING: No model predictions found!")
            print("This means no models were successfully loaded or all models failed during inference.")
            print("\nPlease check:")
            print("1. Are your model files valid .pth files?")
            print("2. Do the filenames contain 'swin' or 'efficientnet'?")
            print("3. Check the error messages above for clues.")
            return
        
        # Print separate tables for each aggregation method if using 'both'
        if self.aggregation == 'both':
            for method in ['majority_voting', 'mean_logits']:
                method_name = "MAJORITY VOTING" if method == 'majority_voting' else "MEAN LOGITS"
                print(f"\n{method_name} RESULTS:")
                print("-" * 120)
                
                # Print header
                print(f"{'Image':<30} | {'Size':<12} | ", end="")
                for model_name in model_names:
                    print(f"{model_name[:15]:<17} | ", end="")
                print()
                print("-" * 120)
                
                # Print each image's results
                for result in results:
                    image_name = Path(result['image_path']).name[:28]
                    image_size = f"{result['image_size'][0]}x{result['image_size'][1]}"
                    
                    print(f"{image_name:<30} | {image_size:<12} | ", end="")
                    
                    for model_name in model_names:
                        model_result = result['models'].get(model_name, {})
                        
                        if 'error' in model_result:
                            print(f"{'ERROR':<17} | ", end="")
                        elif method in model_result:
                            method_result = model_result[method]
                            pred = method_result['predicted_class'][:4].upper()  # AUTH or IMIT
                            conf = method_result['confidence']
                            display = f"{pred} ({conf:.3f})"
                            print(f"{display:<17} | ", end="")
                        else:
                            print(f"{'N/A':<17} | ", end="")
                    print()
                print()
        else:
            # Single aggregation method table
            # Print header
            print(f"\n{'Image':<30} | {'Size':<12} | ", end="")
            for model_name in model_names:
                print(f"{model_name[:15]:<17} | ", end="")
            print()
            print("-" * 120)
            
            # Print each image's results
            for result in results:
                image_name = Path(result['image_path']).name[:28]
                image_size = f"{result['image_size'][0]}x{result['image_size'][1]}"
                
                print(f"{image_name:<30} | {image_size:<12} | ", end="")
                
                for model_name in model_names:
                    model_result = result['models'].get(model_name, {})
                    
                    if 'error' in model_result:
                        print(f"{'ERROR':<17} | ", end="")
                    else:
                        pred = model_result['predicted_class'][:4].upper()  # AUTH or IMIT
                        conf = model_result['confidence']
                        display = f"{pred} ({conf:.3f})"
                        print(f"{display:<17} | ", end="")
                print()
        
        print("="*120)
    
    def predict_folder(self, folder_path, output_file=None):
        """Run inference on all images in a folder."""
        folder_path = Path(folder_path)
        
        # Find image files (case-insensitive, no duplicates)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.jp2'}
        image_files_set = set()
        
        # Get all files in directory
        for file_path in folder_path.iterdir():
            if file_path.is_file():
                # Check if extension matches (case-insensitive)
                if file_path.suffix.lower() in image_extensions:
                    image_files_set.add(file_path)
        
        # Convert to sorted list
        image_files = sorted(list(image_files_set))
        
        print(f"Found {len(image_files)} images in {folder_path}")
        print("="*80)
        
        if not image_files:
            print("No images found!")
            return []
        
        all_results = []
        
        for i, image_path in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] {image_path.name}")
            print("-" * 60)
            
            result = self.predict_image(image_path)
            if result:
                all_results.append(result)
        
        # Display summary table
        self._print_summary_table(all_results)
        
        # Save results if requested
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"\nResults saved to: {output_path}")
        
        return all_results

def main():
    parser = argparse.ArgumentParser(
        description='Van Gogh Authentication Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Aggregation Methods:
  majority  - Majority voting across patches (default)
  mean      - Mean logits aggregation across patches
  both      - Compute and display both methods

Examples:
  # Run with majority voting (default)
  python van_gogh_inference.py --models models/ --images test_images/

  # Run with mean logits aggregation
  python van_gogh_inference.py --models models/ --images test_images/ --aggregation mean

  # Run both methods for comparison
  python van_gogh_inference.py --models models/ --images test_images/ --aggregation both
        """
    )
    parser.add_argument('--models', nargs='+', required=True,
                       help='Paths to model files (.pth) or directories containing models')
    parser.add_argument('--images', type=str, required=True,
                       help='Path to image file or directory containing images')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file to save results')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda, mps, cpu)')
    parser.add_argument('--target_size', type=int, default=256,
                       help='Target image size')
    parser.add_argument('--aggregation', type=str, default='majority',
                       choices=['majority', 'mean', 'both'],
                       help='Aggregation method for multi-patch predictions (default: majority)')
    
    args = parser.parse_args()
    
    try:
        print("Initializing Van Gogh Inference System...")
        
        # Initialize inference system
        inferencer = VanGoghInference(
            model_paths=args.models,
            device=args.device,
            target_size=args.target_size,
            aggregation=args.aggregation
        )
        
        # Check input type
        input_path = Path(args.images)
        
        if input_path.is_file():
            # Single image
            result = inferencer.predict_image(input_path)
            if args.output and result:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump([result], f, indent=2)
                print(f"\nResults saved to: {output_path}")
                
        elif input_path.is_dir():
            # Directory of images
            results = inferencer.predict_folder(input_path, args.output)
            
        else:
            raise ValueError(f"Input path does not exist: {input_path}")
        
        print("\nInference complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())