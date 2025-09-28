import torch
import torch.nn as nn
import torchvision.transforms as transforms
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import numpy as np
import io
import base64
import os
from datetime import datetime

app = FastAPI(title="Trunojoyo AI")

# Mount static folder
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Template rendering
templates = Jinja2Templates(directory="app/templates")
if os.getenv("ENVIRONMENT") == "production":
    app.root_path = "/your-subdirectory"

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Model configuration
LATENT_DIM = 128
IMAGE_SIZE = 64
CHANNELS = 3

class Generator(nn.Module):
    def __init__(self, z_dim=128, img_channels=3, g_channels=64):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Input: z_dim x 1 x 1 -> g_channels*8 x 4 x 4
            nn.ConvTranspose2d(z_dim, g_channels * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(g_channels * 8),
            nn.ReLU(True),

            # g_channels*8 x 4 x 4 -> g_channels*4 x 8 x 8
            nn.ConvTranspose2d(g_channels * 8, g_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_channels * 4),
            nn.ReLU(True),

            # g_channels*4 x 8 x 8 -> g_channels*2 x 16 x 16
            nn.ConvTranspose2d(g_channels * 4, g_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_channels * 2),
            nn.ReLU(True),

            # g_channels*2 x 16 x 16 -> g_channels x 32 x 32
            nn.ConvTranspose2d(g_channels * 2, g_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_channels),
            nn.ReLU(True),

            # g_channels x 32 x 32 -> img_channels x 64 x 64
            nn.ConvTranspose2d(g_channels, img_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)

# Global model variable
generator = None

def inspect_model():
    """Inspect the saved model structure"""
    try:
        model_path = "app/models/generator.pth"
        checkpoint = torch.load(model_path, map_location='cpu')
        
        print("🔍 Model checkpoint keys:")
        if isinstance(checkpoint, dict):
            for key in checkpoint.keys():
                print(f"  - {key}")
                
            # If it's a full checkpoint
            if 'generator_state_dict' in checkpoint:
                state_dict = checkpoint['generator_state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
            
        print("\n🏗️ Model layers:")
        for layer_name in state_dict.keys():
            print(f"  - {layer_name}: {state_dict[layer_name].shape}")
            
    except Exception as e:
        print(f"❌ Error inspecting model: {e}")

def load_model():
    """Load the trained model"""
    global generator
    try:
        model_path = "app/models/generator.pth"
        
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return False
            
        print("🔍 Inspecting model structure...")
        inspect_model()
        
        # Initialize generator with correct z_dim=128
        generator = Generator(z_dim=LATENT_DIM, img_channels=CHANNELS, g_channels=64).to(device)
        
        # Load model weights
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'generator_state_dict' in checkpoint:
                generator.load_state_dict(checkpoint['generator_state_dict'])
                print("📦 Loaded from 'generator_state_dict'")
            elif 'model_state_dict' in checkpoint:
                generator.load_state_dict(checkpoint['model_state_dict'])
                print("📦 Loaded from 'model_state_dict'")
            elif 'generator' in checkpoint:
                generator.load_state_dict(checkpoint['generator'])
                print("📦 Loaded from 'generator'")
            else:
                generator.load_state_dict(checkpoint)
                print("📦 Loaded from direct state dict")
        else:
            generator.load_state_dict(checkpoint)
            print("📦 Loaded from direct tensor")
        
        generator.eval()
        print("✅ Model loaded successfully!")
        print(f"📊 Model parameters: {sum(p.numel() for p in generator.parameters()):,}")
        print(f"🎯 Model device: {next(generator.parameters()).device}")
        
        # Test model with dummy input
        with torch.no_grad():
            test_noise = torch.randn(1, LATENT_DIM, 1, 1, device=device)
            test_output = generator(test_noise)
            print(f"🧪 Test output shape: {test_output.shape}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# Load model on startup
load_model()

def tensor_to_base64(tensor):
    """Convert tensor to base64 image string"""
    try:
        # Denormalize from [-1, 1] to [0, 1]
        tensor = (tensor + 1) / 2
        
        # Clamp values
        tensor = torch.clamp(tensor, 0, 1)
        
        # Convert to PIL Image
        transform = transforms.ToPILImage()
        image = transform(tensor.cpu())
        
        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{img_base64}"
    except Exception as e:
        print(f"Error converting tensor to base64: {str(e)}")
        return None

def tensor_to_downloadable(tensor, filename):
    """Convert tensor to downloadable base64 with filename"""
    try:
        # Denormalize from [-1, 1] to [0, 1]
        tensor = (tensor + 1) / 2
        tensor = torch.clamp(tensor, 0, 1)
        
        # Convert to PIL Image
        transform = transforms.ToPILImage()
        image = transform(tensor.cpu())
        
        # Convert to base64 with higher quality
        buffer = io.BytesIO()
        image.save(buffer, format='PNG', optimize=True)
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return {
            "data": f"data:image/png;base64,{img_base64}",
            "filename": filename
        }
    except Exception as e:
        print(f"Error converting tensor to downloadable: {str(e)}")
        return None

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/penelitian/batik", response_class=HTMLResponse)
async def batik_generation(request: Request):
    return templates.TemplateResponse("penelitian/batik.html", {"request": request})

@app.get("/penelitian/batik.html", response_class=HTMLResponse)
async def batik_generation_html(request: Request):
    return templates.TemplateResponse("penelitian/batik.html", {"request": request})

@app.post("/api/generate_batik")
async def generate_batik(request_data: dict):
    """Generate batik images using the trained model - Temporary only, no file saving"""
    try:
        if generator is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Get parameters from request - dengan default values
        batch_size = request_data.get('batch_size', 8)  # Default 8
        noise_dim = request_data.get('noise_dim', 128)  # Default 128
        seed = request_data.get('seed', None)
        
        # Validate parameters
        batch_size = max(1, min(int(batch_size), 8))  # Max 8 images
        noise_dim = int(noise_dim)  # Should be 128
        
        # Set seed if provided
        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed(int(seed))
        
        # Generate noise with specified dimensions
        noise = torch.randn(batch_size, noise_dim, 1, 1, device=device)
        print(f"🎲 Generated noise shape: {noise.shape}")
        
        # Generate images
        with torch.no_grad():
            generated_images = generator(noise)
            print(f"🎨 Generated images shape: {generated_images.shape}")
        
        # Convert to base64 images (temporary display + download)
        results = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for i, img_tensor in enumerate(generated_images):
            # Convert to base64 for immediate display
            base64_img = tensor_to_base64(img_tensor)
            
            # Create downloadable version with filename
            filename = f"batik_generated_{timestamp}_{i+1}.png"
            download_data = tensor_to_downloadable(img_tensor, filename)
            
            if base64_img and download_data:
                results.append({
                    "image_base64": base64_img,              # For display
                    "download_data": download_data["data"],  # For download
                    "filename": download_data["filename"],   # Filename for download
                    "image_id": f"img_{i+1}"                # Unique identifier
                })
        
        print(f"✅ Generated {len(results)} temporary batik images")
        
        return {
            "status": "success", 
            "message": f"Generated {len(results)} batik images successfully (temporary)",
            "images": results,
            "parameters": {
                "batch_size": batch_size,
                "noise_dim": noise_dim,
                "seed": seed,
                "device": str(device)
            },
            "note": "Images are temporary and not saved to server storage"
        }
        
    except Exception as e:
        print(f"Error generating batik: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.get("/api/model_status")
async def model_status():
    """Check if model is loaded and ready"""
    return {
        "model_loaded": generator is not None,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "model_parameters": sum(p.numel() for p in generator.parameters()) if generator else 0,
        "latent_dim": LATENT_DIM,
        "storage_mode": "temporary_only"  # Indicate no file storage
    }

@app.post("/api/reload_model")
async def reload_model():
    """Reload the model"""
    success = load_model()
    return {
        "status": "success" if success else "error",
        "message": "Model reloaded successfully" if success else "Failed to reload model"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)