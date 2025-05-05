import streamlit as st
from PIL import Image
import torch
import numpy as np
import os
import sys
import io
import torchvision.utils as vutils
from torchvision import transforms
import torch.nn as nn

# Set path for imports
sys.path.append(os.path.abspath('../mambast'))
sys.path.append(os.path.abspath('../steganography'))

# Import models
from util.utils import load_pretrained
from stega_models.HidingUNet import UnetGenerator
from stega_models.RevealNet import RevealNet

# Set page configuration for a cleaner look
st.set_page_config(
    page_title="Neural Style Transfer with Content Preservation",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for a cleaner interface
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    .stButton button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 0.5rem;
        border-radius: 4px;
    }
    h1, h2, h3 {
        margin-bottom: 0.5rem;
    }
    .caption {
        font-size: 0.8rem;
        color: #666;
        text-align: center;
        margin-top: 0.2rem;
    }
    .result-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        margin-top: 1rem;
    }
    .download-btn {
        margin-top: 1rem;
    }
    .comparison-container {
        display: flex;
        justify-content: space-between;
        margin-top: 1rem;
    }
    .mode-selector {
        background-color: #f1f3f5;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f1f3f5;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50 !important;
        color: white !important;
    }
    .step-container {
        border-left: 3px solid #4CAF50;
        padding-left: 15px;
        margin-bottom: 20px;
    }
    .flow-arrow {
        text-align: center;
        font-size: 24px;
        color: #888;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_image(image, img_size):
    """Preprocess an image for the models"""
    image = image.convert("RGB").resize((img_size, img_size))
    return transforms.ToTensor()(image).unsqueeze(0).to(device)

@st.cache_resource
def load_style_transfer_model():
    """Load the style transfer model"""
    class Args:
        mamba_path = '../checkpoints/mamba_iter_160000.pth'
        embedding_path = '../checkpoints/embedding_iter_160000.pth'
        decoder_path = '../checkpoints/decoder_iter_160000.pth'
        vgg = '../checkpoints/vgg_normalised.pth'
        d_state = 16
        img_size = 256
        use_pos_embed = False
        rnd_style = False
    model = load_pretrained(Args()).to(device).eval()
    return model, Args()

@st.cache_resource
def load_steganography_models():
    """Load the hiding and reveal networks"""
    Hnet = UnetGenerator(input_nc=6, output_nc=3, num_downs=7, output_function=nn.Sigmoid).to(device)
    try:
        hnet_checkpoint = torch.load("../checkpoints/netH_epoch_73,sumloss=0.000447,Hloss=0.000258.pth", 
                                   map_location=device)
        Hnet.load_state_dict(hnet_checkpoint)
    except FileNotFoundError:
        st.warning("Hiding network checkpoint not found. Using uninitialized model.")
    Hnet.eval()

    Rnet = RevealNet(output_function=nn.Sigmoid).to(device)
    try:
        rnet_checkpoint = torch.load("../checkpoints/netR_epoch_73,sumloss=0.000447,Rloss=0.000252.pth", 
                                   map_location=device)
        Rnet.load_state_dict(rnet_checkpoint)
    except FileNotFoundError:
        st.warning("Reveal network checkpoint not found. Using uninitialized model.")
    Rnet.eval()
    
    return Hnet, Rnet

def tensor_to_image(tensor):
    """Convert a tensor to a PIL Image"""
    # Remove batch dimension and move to CPU
    img = tensor.squeeze(0).cpu().detach()
    # Convert to numpy and transpose to (H, W, C)
    img = img.numpy().transpose(1, 2, 0)
    # Clip to [0, 1] and convert to uint8
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)

def run_serial_style_transfer(content_tensor, style_tensors, model):
    """Run serial style transfer"""
    outputs = []
    current_content = content_tensor.clone()
    
    for idx, style_tensor in enumerate(style_tensors):
        with torch.no_grad():
            output_tensor, *_ = model(current_content, style_tensor)
        
        output_img = tensor_to_image(output_tensor)
        outputs.append(output_img)
        
        # Update for next iteration
        current_content = preprocess_image(output_img, 256)
    
    return outputs

def run_content_preserving_style_transfer(content_tensor, style_tensors, model, Hnet, Rnet):
    """Run style transfer with content preservation using steganography"""
    styled_tensors = []
    container_tensors = []
    revealed_tensors = []
    
    # Original content tensor for the first iteration
    current_content = content_tensor.clone()
    
    for idx, style_tensor in enumerate(style_tensors):
        # Step 1: Style transfer with current content
        with torch.no_grad():
            styled_tensor, *_ = model(current_content, style_tensor)
        styled_tensors.append(styled_tensor)
        
        # Step 2: Hide original content in styled image
        concat_tensor = torch.cat((styled_tensor, current_content), dim=1)
        with torch.no_grad():
            container_tensor = Hnet(concat_tensor).clamp(0, 1)
        container_tensors.append(container_tensor)
        
        # Step 3: Reveal content from container
        with torch.no_grad():
            revealed_tensor = Rnet(container_tensor).clamp(0, 1)
        revealed_tensors.append(revealed_tensor)
        
        # Step 4: Use revealed content for next style
        current_content = revealed_tensor.clone()
    
    return styled_tensors, container_tensors, revealed_tensors

# Header with minimalist design
st.markdown("<h1 style='text-align: center;'>Neural Style Transfer Studio</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; margin-bottom: 1rem;'>Transform images with styles while preserving content</p>", unsafe_allow_html=True)

# Mode selection
with st.sidebar:
    st.markdown("### Mode Selection")
    mode = st.radio(
        "Choose Transfer Mode:",
        ["Serial Style Transfer", "Content Preserving Style Transfer", "Style Transfer", "Style Transfer with Steganography"],
        help="Serial applies each style sequentially. Content Preserving uses steganography to retain original content."
    )
    
    st.markdown("### About the Modes")
    st.markdown("""
    **Serial Style Transfer**: 
    Each style is applied sequentially to the output of the previous style transfer.
    
    **Content Preserving**:
    Uses the following workflow for each style:
    1. Apply style to content
    2. Hide original content in styled image
    3. Reveal content from container image
    4. Use revealed content for next style application
    """)

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Content Image")
    content_file = st.file_uploader("Upload content image", type=["jpg", "png", "jpeg"], key="content", label_visibility="collapsed")
    
with col2:
    st.markdown("### Style Images")
    style_files = st.file_uploader("Upload style images (multiple allowed)", type=["jpg", "png", "jpeg"], 
                                   accept_multiple_files=True, key="style", label_visibility="collapsed")
has_steganography = st.checkbox("🧬 Has Steganography?", value=True)
# Add a horizontal rule
st.markdown("---")

# Display and process images if uploaded
if content_file and style_files:
    # Preview row
    st.markdown("### Image Preview")
    preview_cols = st.columns([1] + [1] * min(3, len(style_files)))
    
    content_img = Image.open(content_file).resize((256, 256))
    style_imgs = [Image.open(sf).resize((256, 256)) for sf in style_files]
    
    preview_cols[0].image(content_img, caption="Content Image", use_container_width=True)
    
    for i, style_img in enumerate(style_imgs[:3]):
        preview_cols[i+1].image(style_img, caption=f"Style {i+1}", use_container_width=True)
    
    if len(style_imgs) > 3:
        st.caption(f"+ {len(style_imgs) - 3} more style images")
    
    # Options row
    options_col1, options_col2 = st.columns([1, 1])
    
    with options_col1:
        st.button("✨ Generate Stylized Images", key="apply", type="primary")
    
    # Processing logic
    if st.session_state.get("apply", False):
        st.markdown("---")
        
        if mode == "Serial Style Transfer":
            st.markdown("### Serial Style Transfer Results")
            
            with st.spinner("Applying serial style transfer..."):
                model, args = load_style_transfer_model()
                content_tensor = preprocess_image(content_img, args.img_size)
                style_tensors = [preprocess_image(img, args.img_size) for img in style_imgs]
                
                progress_bar = st.progress(0)
                styled_outputs = run_serial_style_transfer(content_tensor, style_tensors, model)
                
                # Display results
                st.markdown("<div class='result-container'>", unsafe_allow_html=True)
                
                for i, output in enumerate(styled_outputs):
                    cols = st.columns(3)
                    cols[0].image(style_imgs[i], caption=f"Style {i+1}", use_container_width=True)
                    
                    # Display the previous result if not the first style
                    if i > 0:
                        cols[1].image(styled_outputs[i-1], caption=f"Result {i}", use_container_width=True)
                    else:
                        cols[1].image(content_img, caption="Original Content", use_container_width=True)
                    
                    cols[2].image(output, caption=f"Result {i+1}", use_container_width=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Final result
                final_result = styled_outputs[-1]
                    
        elif mode == "Content Preserving Style Transfer":  # Content Preserving Style Transfer
            st.markdown("### Content Preserving Style Transfer Results")
            
            with st.spinner("Applying content preserving style transfer..."):
                model, args = load_style_transfer_model()
                Hnet, Rnet = load_steganography_models()
                
                content_tensor = preprocess_image(content_img, args.img_size)
                style_tensors = [preprocess_image(img, args.img_size) for img in style_imgs]
                
                progress_bar = st.progress(0)
                styled_tensors, container_tensors, revealed_tensors = run_content_preserving_style_transfer(
                    content_tensor, style_tensors, model, Hnet, Rnet
                )
                
                # Convert tensors to images
                styled_images = [tensor_to_image(tensor) for tensor in styled_tensors]
                container_images = [tensor_to_image(tensor) for tensor in container_tensors]
                revealed_images = [tensor_to_image(tensor) for tensor in revealed_tensors]
                
                # Display the complete flow for each style
                for i in range(len(styled_images)):
                    st.markdown(f"## Style {i+1} Processing Flow")
                    
                    # Step 1: Style Transfer
                    st.markdown(f"<div class='step-container'>", unsafe_allow_html=True)
                    st.markdown("### Step 1: Style Transfer")
                    cols = st.columns(4)
                    
                    # Show input content (original or revealed from previous step)
                    if i == 0:
                        cols[0].image(content_img, caption="Original Content", use_container_width=True)
                        input_content = "Original Content"
                    else:
                        cols[0].image(revealed_images[i-1], caption=f"Revealed Content {i}", use_container_width=True)
                        input_content = f"Revealed Content {i}"
                    
                    cols[1].image(style_imgs[i], caption=f"Style {i+1}", use_container_width=True)
                    
                    # Arrow
                    cols[2].markdown("<div class='flow-arrow'>➡️</div>", unsafe_allow_html=True)
                    
                    # Styled output
                    cols[3].image(styled_images[i], caption=f"Stylized Image {i+1}", use_container_width=True)
                    
                    st.markdown(f"**Process**: {input_content} + Style {i+1} → Stylized Image {i+1}")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Step 2: Hiding
                    st.markdown(f"<div class='step-container'>", unsafe_allow_html=True)
                    st.markdown("### Step 2: Content Hiding")
                    cols = st.columns(4)
                    
                    cols[0].image(styled_images[i], caption=f"Stylized Image {i+1}", use_container_width=True)
                    cols[1].image(content_img, caption="Original Content", use_container_width=True)
                    
                    # Arrow
                    cols[2].markdown("<div class='flow-arrow'>➡️</div>", unsafe_allow_html=True)
                    
                    # Container image
                    cols[3].image(container_images[i], caption=f"Container Image {i+1}", use_container_width=True)
                    
                    st.markdown(f"**Process**: Stylized Image {i+1} + Original Content → Container Image {i+1} (original content hidden inside)")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Step 3: Revealing
                    st.markdown(f"<div class='step-container'>", unsafe_allow_html=True)
                    st.markdown("### Step 3: Content Revealing")
                    cols = st.columns(3)
                    
                    cols[0].image(container_images[i], caption=f"Container Image {i+1}", use_container_width=True)
                    
                    # Arrow
                    cols[1].markdown("<div class='flow-arrow'>➡️</div>", unsafe_allow_html=True)
                    
                    # Revealed content
                    cols[2].image(revealed_images[i], caption=f"Revealed Content {i+1}", use_container_width=True)
                    
                    st.markdown(f"**Process**: Container Image {i+1} → Revealed Content {i+1} (will be used for next style)")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Step 4: Next Style (if not the last one)
                    if i < len(styled_images) - 1:
                        st.markdown("<div class='flow-arrow'>⬇️</div>", unsafe_allow_html=True)
                        st.markdown(f"**Next Step**: Revealed Content {i+1} will be used with Style {i+2}")
                    
                    st.markdown("---")
                
                # Final result
                final_result = container_images[-1]
        elif mode == "Style Transfer":
            st.markdown("### Style Transfer Results")
            with st.spinner("Applying style transfer..."):
                model, args = load_style_transfer_model()
                content_tensor = preprocess_image(content_img, args.img_size)
                style_tensor = preprocess_image(style_imgs[0], args.img_size)
                
                with torch.no_grad():
                    output_tensor, *_ = model(content_tensor, style_tensor)
                
                final_result = tensor_to_image(output_tensor)
                st.image(final_result, caption="Final Output", use_container_width=True)
                st.markdown("**Process**: Original Content + Style 1 → Final Output")
                st.markdown("---")  
        else:
            st.markdown("### Style Transfer with Steganography Results")
            with st.spinner("Applying style transfer with steganography..."):
                model, args = load_style_transfer_model()
                Hnet, Rnet = load_steganography_models()
                
                content_tensor = preprocess_image(content_img, args.img_size)
                if has_steganography:
                    with torch.no_grad():
                        content_tensor = Rnet(content_tensor).clamp(0, 1)
                style_tensor = preprocess_image(style_imgs[0], args.img_size)
                
                # Step 1: Style transfer
                with torch.no_grad():
                    styled_tensor, *_ = model(content_tensor, style_tensor)
                
                # Step 2: Hide original content in styled image
                concat_tensor = torch.cat((styled_tensor, content_tensor), dim=1)
                with torch.no_grad():
                    container_tensor = Hnet(concat_tensor).clamp(0, 1)
                if has_steganography:
                    revealed_image = tensor_to_image(content_tensor)
                final_result = tensor_to_image(container_tensor)
                st.image(final_result, caption="Final Output", use_container_width=True)
                st.markdown("**Process**: Original Content + Style 1 → Container Image (original content hidden inside)")
                st.markdown("---")
        # Save the final image to a bytes buffer
        buf = io.BytesIO()
        final_result.save(buf, format="PNG")
        byte_im = buf.getvalue()
        
        # Final results comparison section
        st.markdown("## Final Results Comparison")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.image(content_img, caption="Original Content", use_container_width=True)
        
        with col2:
            st.image(final_result, caption="Final Output", use_container_width=True)
            
        with col3:
            st.markdown("<div class='download-btn'>", unsafe_allow_html=True)
            st.download_button(
                "📥 Download Result",
                data=byte_im,
                file_name=f"style_transfer_{mode.replace(' ', '_').lower()}.png",
                mime="image/png"
            )
            st.markdown("</div>", unsafe_allow_html=True)
        
        if mode == "Content Preserving Style Transfer":
            st.markdown("### Content Preservation Quality")
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(content_img, caption="Original Content", use_container_width=True)
            
            with col2:
                st.image(revealed_images[-1], caption="Final Revealed Content", use_container_width=True)
            
            # Calculate PSNR between original content and final revealed content
            orig_tensor = preprocess_image(content_img, 256)
            revealed_tensor = preprocess_image(revealed_images[-1], 256)
            
            mse = torch.mean((orig_tensor - revealed_tensor) ** 2)
            psnr = 10 * torch.log10(1 / mse)
            
            st.markdown(f"**Content Preservation Quality (PSNR)**: {psnr.item():.2f} dB")
            st.markdown("Higher PSNR values indicate better content preservation.")
        if mode == "Style Transfer with Steganography" and has_steganography:
            st.markdown("### Steganography Quality")
            col1, col2 = st.columns(2)
            with col1:
                st.image(content_img, caption="Original Content", use_container_width=True)
            with col2:
                st.image(revealed_image, caption="Revealed Image", use_container_width=True)

else:
    # Placeholder instructions when no images are uploaded
    st.info("👆 Please upload a content image and at least one style image to begin")
    
    st.markdown("### Content Preserving Style Transfer Flow")
    st.markdown("""
    The content preserving mode follows this workflow for each style:
    
    1. **Style Transfer**: Apply the style to the content image (original content or revealed content from previous step)
    2. **Content Hiding**: Hide the original content inside the stylized image using steganography
    3. **Content Revealing**: Extract the content from the container image
    4. **Next Style**: Use the revealed content as input for the next style transfer
    
    This approach helps maintain content integrity through multiple style transfers.
    """)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #888; font-size: 0.8rem;'>Powered by Neural Style Transfer with Steganography</p>", unsafe_allow_html=True)