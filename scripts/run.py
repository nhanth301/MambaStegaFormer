import streamlit as st
from PIL import Image
import torch
import numpy as np
import os
import sys
from torchvision import transforms
import io

sys.path.append(os.path.abspath('../MambaST'))
from util.utils import load_pretrained

# Set page configuration for a cleaner look
st.set_page_config(
    page_title="Neural Style Transfer",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="collapsed"
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
</style>
""", unsafe_allow_html=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_image(image, img_size):
    image = image.convert("RGB").resize((img_size, img_size))
    return transforms.ToTensor()(image).unsqueeze(0).to(device)

@st.cache_resource
def load_model():
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

# Header with minimalist design
st.markdown("<h1 style='text-align: center;'>Neural Style Transfer Studio</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; margin-bottom: 2rem;'>Transform your images using multiple artistic styles</p>", unsafe_allow_html=True)

# Use columns for input layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Content Image")
    content_file = st.file_uploader("Upload content image", type=["jpg", "png", "jpeg"], key="content", label_visibility="collapsed")
    
with col2:
    st.markdown("### Style Images")
    style_files = st.file_uploader("Upload style images (multiple allowed)", type=["jpg", "png", "jpeg"], 
                                   accept_multiple_files=True, key="style", label_visibility="collapsed")

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
    options_col1, options_col2, options_col3 = st.columns([1, 1, 1])
    
    with options_col1:
        st.button("✨ Generate Stylized Images", key="apply", type="primary")
    
    # Processing logic
    if st.session_state.get("apply", False):
        st.markdown("---")
        st.markdown("### Processing Results")
        
        with st.spinner("Applying style transfer..."):
            model, args = load_model()
            content_tensor = preprocess_image(content_img, args.img_size)
            outputs = []
            
            progress_bar = st.progress(0)
            
            for idx, style_img in enumerate(style_imgs):
                style_tensor = preprocess_image(style_img, args.img_size)
                with torch.no_grad():
                    output_tensor, *_ = model(content_tensor, style_tensor)
                
                # Sử dụng:
                output_np = output_tensor.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)
                # Chuẩn hóa giá trị về khoảng [0, 1] nếu cần
                output_np = np.clip(output_np, 0, 1)
                output_img = Image.fromarray((output_np * 255).astype(np.uint8))
                outputs.append((style_img, output_img))
                
                # Update for next iteration if doing sequential style transfer
                content_tensor = preprocess_image(output_img, args.img_size)
                
                # Update progress
                progress_bar.progress((idx + 1) / len(style_imgs))
        
        # Results display in a grid
        st.markdown("<div class='result-container'>", unsafe_allow_html=True)
        
        # For even tighter layout, show in groups of 4
        for i in range(0, len(outputs), 2):
            cols = st.columns(4)
            
            # First pair
            cols[0].image(outputs[i][0], caption=f"Style {i+1}", use_container_width=True)
            cols[1].image(outputs[i][1], caption=f"Result {i+1}", use_container_width=True)
            
            # Second pair (if exists)
            if i+1 < len(outputs):
                cols[2].image(outputs[i+1][0], caption=f"Style {i+2}", use_container_width=True)
                cols[3].image(outputs[i+1][1], caption=f"Result {i+2}", use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Export final result
        final_result = outputs[-1][1]
        
        # Save the final image to a bytes buffer
        buf = io.BytesIO()
        final_result.save(buf, format="JPEG")
        byte_im = buf.getvalue()
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.image(final_result, caption="Final Output", width=400)
        
        with col2:
            st.markdown("<div class='download-btn'>", unsafe_allow_html=True)
            st.download_button(
                "📥 Download Result",
                data=byte_im,
                file_name="style_transfer_result.jpg",
                mime="image/jpeg"
            )
            st.markdown("</div>", unsafe_allow_html=True)
else:
    # Placeholder instructions when no images are uploaded
    st.info("👆 Please upload a content image and at least one style image to begin")
    
    # Display sample images if available (commented out for now)
    # st.markdown("### Sample Results")
    # sample_cols = st.columns(3)
    # sample_cols[0].image("sample_content.jpg", caption="Sample Content")
    # sample_cols[1].image("sample_style.jpg", caption="Sample Style")
    # sample_cols[2].image("sample_result.jpg", caption="Sample Result")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #888; font-size: 0.8rem;'>Powered by MambaST Neural Style Transfer</p>", unsafe_allow_html=True)