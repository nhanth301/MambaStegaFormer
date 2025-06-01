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

# Set page configuration
st.set_page_config(
    page_title="Neural Style Transfer & Steganography Studio",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern design
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .header-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 8px;
        margin-bottom: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 8px;
        color: #495057;
        font-weight: 600;
        font-size: 1rem;
        padding: 12px 24px;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white !important;
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3);
    }
    
    /* Upload area styling */
    .upload-section {
        # background: #f8f9fa;
        # border-radius: 15px;
        # padding: 2rem;
        # margin-bottom: 2rem;
        # border: 2px dashed #dee2e6;
        # transition: all 0.3s ease;
    }
    
    .upload-section:hover {
        border-color: #4facfe;
        background: rgba(79, 172, 254, 0.05);
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 6px 20px rgba(40, 167, 69, 0.3);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(40, 167, 69, 0.4);
    }
    
    /* Results section */
    .results-container {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        margin-top: 2rem;
    }
    
    .results-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #495057;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    /* Info cards */
    .info-card {
        background: linear-gradient(135deg, #17a2b8 0%, #6610f2 100%);
        color: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 2rem;
    }
    
    .info-title {
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .info-description {
        line-height: 1.8;
        opacity: 0.95;
        font-size: 1rem;
    }
    
    /* Process flow */
    .process-flow {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 20px;
        margin: 2rem 0;
        flex-wrap: wrap;
    }
    
    .flow-step {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        text-align: center;
        min-width: 120px;
        font-weight: 600;
        color: #495057;
    }
    
    .flow-arrow {
        font-size: 1.5rem;
        color: #4facfe;
        font-weight: bold;
    }
    
    /* Image styling */
    .image-container {
        text-align: center;
        margin: 1rem 0;
    }
    
    .image-label {
        font-weight: 600;
        color: #495057;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
    }
    
    /* Success/Error messages */
    .stSuccess {
        border-radius: 10px;
        border-left: 4px solid #28a745;
    }
    
    .stError {
        border-radius: 10px;
        border-left: 4px solid #dc3545;
    }
    
    .stWarning {
        border-radius: 10px;
        border-left: 4px solid #ffc107;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_image(image, img_size=512):
    """Preprocess an image for the models"""
    image = image.convert("RGB").resize((img_size, img_size))
    return transforms.ToTensor()(image).unsqueeze(0).to(device)

@st.cache_resource
def load_style_transfer_model():
    """Load the style transfer model"""
    class Args:
        mamba_path = '../checkpoints/mamba_iter_65000.pth'
        embedding_path = '../checkpoints/embedding_iter_160000.pth'
        decoder_path = '../checkpoints/decoder_iter_160000.pth'
        vgg = '../checkpoints/vgg_normalised.pth'
        d_state = 16
        img_size = 512
        use_pos_embed = True
        rnd_style = True
    
    try:
        model = load_pretrained(Args()).to(device).eval()
        return model, Args()
    except Exception as e:
        st.error(f"Failed to load style transfer model: {str(e)}")
        return None, None

@st.cache_resource
def load_steganography_models():
    """Load the hiding and reveal networks"""
    try:
        # Load Hiding Network
        Hnet = UnetGenerator(input_nc=6, output_nc=3, num_downs=7, output_function=nn.Sigmoid).to(device)
        hnet_checkpoint = torch.load("../checkpoints/netH_epoch_121,sumloss=0.000686,Hloss=0.000388.pth", 
                                   map_location=device)
        Hnet.load_state_dict(hnet_checkpoint)
        Hnet.eval()
        
        # Load Reveal Network
        Rnet = RevealNet(output_function=nn.Sigmoid).to(device)
        rnet_checkpoint = torch.load("../checkpoints/netR_epoch_121,sumloss=0.000686,Rloss=0.000397.pth", 
                                   map_location=device)
        Rnet.load_state_dict(rnet_checkpoint)
        Rnet.eval()
        
        return Hnet, Rnet
    except Exception as e:
        st.error(f"Failed to load steganography models: {str(e)}")
        return None, None

def tensor_to_image(tensor):
    """Convert a tensor to a PIL Image"""
    img = tensor.squeeze(0).cpu().detach()
    img = img.numpy().transpose(1, 2, 0)
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)

def image_to_bytes(image):
    """Convert PIL image to bytes for download"""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()

# Header
st.markdown("""
<div class="header-container">
    <div class="header-title">🎨 Neural Style Transfer & Steganography Studio</div>
    <div class="header-subtitle">Advanced AI-powered image processing with style transfer and steganography</div>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 🛠️ System Status")
    
    # Check device
    device_status = "🟢 GPU Available" if torch.cuda.is_available() else "🟡 CPU Only"
    st.markdown(f"**Device:** {device_status}")
    
    # Model loading status
    style_model, args = load_style_transfer_model()
    Hnet, Rnet = load_steganography_models()
    
    if style_model is not None:
        st.markdown("**Style Transfer Model:** 🟢 Loaded")
    else:
        st.markdown("**Style Transfer Model:** 🔴 Failed")
    
    if Hnet is not None and Rnet is not None:
        st.markdown("**Steganography Models:** 🟢 Loaded")
    else:
        st.markdown("**Steganography Models:** 🔴 Failed")
    
    st.markdown("---")
    st.markdown("## 📖 Quick Guide")
    st.markdown("""
    **Style Transfer**: Apply artistic styles to images
    
    **Serial Transfer**: Reveal → Style transfer
    
    **Hiding**: Hide secret in cover image
    
    **Revealing**: Extract hidden content
    """)

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎨 Style Transfer", 
    "🔄 Serial Style Transfer", 
    "🔒 Steganography Hiding", 
    "🔓 Steganography Revealing"
])

# Tab 1: Style Transfer
with tab1:
    st.markdown("""
    <div class="info-card">
        <div class="info-title">🎨 Style Transfer</div>
        <div class="info-description">
            Apply artistic styles to your content images. You can optionally use steganography to hide the original content within the stylized result, allowing you to recover the original image later.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown("### 🖼️ Content Image")
        content_file = st.file_uploader("Upload your content image", type=["jpg", "png", "jpeg"], key="content_1")
        if content_file:
            content_img = Image.open(content_file)
            content_img = content_img.resize((512,512))
            st.image(content_img, caption="Content Image", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown("### 🎭 Style Image")
        style_file = st.file_uploader("Upload your style reference", type=["jpg", "png", "jpeg"], key="style_1")
        if style_file:
            style_img = Image.open(style_file)
            style_img = style_img.resize((512,512))
            st.image(style_img, caption="Style Image", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Options
    use_steganography = st.checkbox("🔒 Use Steganography (Hide original content in stylized image)", key="stego_1")
    
    if use_steganography:
        st.info("💡 With steganography enabled, the original content will be hidden inside the stylized image and can be recovered later using the revealing function.")
    
    # Process button
    if st.button("✨ Generate Styled Image", type="primary", key="process_1"):
        if content_file and style_file and style_model is not None:
            with st.spinner("🎨 Processing style transfer..."):
                try:
                    # Preprocess images
                    content_tensor = preprocess_image(content_img, args.img_size)
                    style_tensor = preprocess_image(style_img, args.img_size)
                    
                    # Style transfer
                    with torch.no_grad():
                        styled_tensor, *_ = style_model(content_tensor, style_tensor)
                        styled_tensor = styled_tensor.clamp(0, 1)
                    
                    styled_image = tensor_to_image(styled_tensor)
                    
                    if use_steganography and Hnet is not None:
                        # Hide original content in styled image
                        concat_tensor = torch.cat((styled_tensor, content_tensor), dim=1)
                        with torch.no_grad():
                            container_tensor = Hnet(concat_tensor).clamp(0, 1)
                        final_result = tensor_to_image(container_tensor)
                        result_label = "Styled Image (with hidden content)"
                    else:
                        final_result = styled_image
                        result_label = "Styled Image"
                    
                    # Display results
                    st.markdown('<div class="results-container">', unsafe_allow_html=True)
                    st.markdown('<div class="results-title">🎯 Results</div>', unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.image(content_img, caption="Original Content", use_container_width=True)
                    with col2:
                        st.image(style_img, caption="Style Reference", use_container_width=True)
                    with col3:
                        st.image(final_result, caption=result_label, use_container_width=True)
                        st.download_button(
                            "📥 Download Result",
                            data=image_to_bytes(final_result),
                            file_name="styled_image.png",
                            mime="image/png"
                        )
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.success("✅ Style transfer completed successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error during processing: {str(e)}")
        else:
            st.warning("⚠️ Please upload both content and style images, and ensure models are loaded.")

# Tab 2: Serial Style Transfer
# Tab 2: Serial Style Transfer
with tab2:
    st.markdown("""
    <div class="info-card">
        <div class="info-title">🔄 Serial Style Transfer</div>
        <div class="info-description">
            First reveal hidden content from the input image using the reveal network, then apply style transfer to the revealed content. Optionally, hide the revealed content back into the stylized result using steganography for later recovery.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Process flow visualization
    st.markdown("""
    <div class="process-flow">
        <div class="flow-step">
            <strong>Step 1</strong><br>
            🔓 Reveal Hidden Content
        </div>
        <div class="flow-arrow">→</div>
        <div class="flow-step">
            <strong>Step 2</strong><br>
            🎨 Apply Style Transfer
        </div>
        <div class="flow-arrow">→</div>
        <div class="flow-step">
            <strong>Step 3</strong><br>
            🔒 (Optional) Hide Content
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown("### 🔓 Content Image (with hidden data)")
        content_file_2 = st.file_uploader("Upload image containing hidden content", type=["jpg", "png", "jpeg"], key="content_2")
        if content_file_2:
            content_img_2 = Image.open(content_file_2)
            st.image(content_img_2, caption="Input Image", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown("### 🎭 Style Image")
        style_file_2 = st.file_uploader("Upload your style reference", type=["jpg", "png", "jpeg"], key="style_2")
        if style_file_2:
            style_img_2 = Image.open(style_file_2)
            st.image(style_img_2, caption="Style Image", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Options
    use_steganography_2 = st.checkbox("🔒 Use Steganography (Hide revealed content in stylized image)", key="stego_2")
    
    if use_steganography_2:
        st.info("💡 With steganography enabled, the revealed content will be hidden inside the stylized image and can be recovered later using the revealing function.")
    
    if st.button("🔄 Process Serial Style Transfer", type="primary", key="process_2"):
        if content_file_2 and style_file_2 and style_model is not None and Rnet is not None:
            with st.spinner("🔄 Processing serial style transfer..."):
                try:
                    # Step 1: Reveal hidden content
                    content_tensor_2 = preprocess_image(content_img_2, args.img_size)
                    with torch.no_grad():
                        revealed_tensor = Rnet(content_tensor_2).clamp(0, 1)
                    revealed_image = tensor_to_image(revealed_tensor)
                    
                    # Step 2: Style transfer on revealed content
                    style_tensor_2 = preprocess_image(style_img_2, args.img_size)
                    with torch.no_grad():
                        styled_tensor, *_ = style_model(revealed_tensor, style_tensor_2)
                        styled_tensor = styled_tensor.clamp(0, 1)
                    styled_image = tensor_to_image(styled_tensor)
                    
                    # Step 3: Optional steganography
                    if use_steganography_2 and Hnet is not None:
                        # Hide revealed content in styled image
                        concat_tensor = torch.cat((styled_tensor, revealed_tensor), dim=1)
                        with torch.no_grad():
                            container_tensor = Hnet(concat_tensor).clamp(0, 1)
                        final_result = tensor_to_image(container_tensor)
                        result_label = "Styled Image (with hidden content)"
                    else:
                        final_result = styled_image
                        result_label = "Styled Image"
                    
                    # Display results
                    st.markdown('<div class="results-container">', unsafe_allow_html=True)
                    st.markdown('<div class="results-title">🎯 Serial Processing Results</div>', unsafe_allow_html=True)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.image(content_img_2, caption="Input Image", use_container_width=True)
                    with col2:
                        st.image(revealed_image, caption="Revealed Content", use_container_width=True)
                    with col3:
                        st.image(style_img_2, caption="Style Reference", use_container_width=True)
                    with col4:
                        st.image(final_result, caption=result_label, use_container_width=True)
                        st.download_button(
                            "📥 Download Result",
                            data=image_to_bytes(final_result),
                            file_name="serial_styled_image.png",
                            mime="image/png"
                        )
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.success("✅ Serial style transfer completed successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error during processing: {str(e)}")
        else:
            st.warning("⚠️ Please upload both images and ensure all models are loaded.")

# Tab 3: Steganography Hiding
with tab3:
    st.markdown("""
    <div class="info-card">
        <div class="info-title">🔒 Steganography Hiding</div>
        <div class="info-description">
            Hide a secret image inside a cover image using deep learning steganography. The resulting container image will visually appear like the cover image but contains the hidden secret image that can be extracted later.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown("### 🖼️ Cover Image")
        st.markdown("*The image that will be visible*")
        cover_file = st.file_uploader("Upload cover image", type=["jpg", "png", "jpeg"], key="cover")
        if cover_file:
            cover_img = Image.open(cover_file)
            st.image(cover_img, caption="Cover Image", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown("### 🤫 Secret Image")
        st.markdown("*The image to be hidden*")
        secret_file = st.file_uploader("Upload secret image", type=["jpg", "png", "jpeg"], key="secret")
        if secret_file:
            secret_img = Image.open(secret_file)
            st.image(secret_img, caption="Secret Image", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("🔒 Hide Secret Image", type="primary", key="process_3"):
        if cover_file and secret_file and Hnet is not None:
            with st.spinner("🔒 Hiding secret image..."):
                try:
                    # Preprocess images
                    cover_tensor = preprocess_image(cover_img, 512)
                    secret_tensor = preprocess_image(secret_img, 512)
                    
                    # Hide secret in cover
                    concat_tensor = torch.cat((cover_tensor, secret_tensor), dim=1)
                    with torch.no_grad():
                        container_tensor = Hnet(concat_tensor).clamp(0, 1)
                    container_image = tensor_to_image(container_tensor)
                    
                    # Display results
                    st.markdown('<div class="results-container">', unsafe_allow_html=True)
                    st.markdown('<div class="results-title">🎯 Hiding Results</div>', unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.image(cover_img, caption="Cover Image", use_container_width=True)
                    with col2:
                        st.image(secret_img, caption="Secret Image", use_container_width=True)
                    with col3:
                        st.image(container_image, caption="Container Image", use_container_width=True)
                        st.download_button(
                            "📥 Download Container",
                            data=image_to_bytes(container_image),
                            file_name="container_image.png",
                            mime="image/png"
                        )
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.success("✅ Secret image hidden successfully!")
                    st.info("💡 The container image looks like the cover image but contains the hidden secret. Use the 'Steganography Revealing' tab to extract the secret.")
                    
                except Exception as e:
                    st.error(f"❌ Error during hiding: {str(e)}")
        else:
            st.warning("⚠️ Please upload both cover and secret images, and ensure hiding model is loaded.")

# Tab 4: Steganography Revealing
with tab4:
    st.markdown("""
    <div class="info-card">
        <div class="info-title">🔓 Steganography Revealing</div>
        <div class="info-description">
            Extract and reveal hidden content from a container image that was created using steganographic hiding techniques. Upload a container image to recover the secret image hidden within it.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown("### 📦 Container Image")
    st.markdown("*Upload the image containing hidden content*")
    container_file = st.file_uploader("Upload container image", type=["jpg", "png", "jpeg"], key="container")
    if container_file:
        container_img = Image.open(container_file)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(container_img, caption="Container Image", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("🔓 Reveal Hidden Content", type="primary", key="process_4"):
        if container_file and Rnet is not None:
            with st.spinner("🔓 Revealing hidden content..."):
                try:
                    # Preprocess container image
                    container_tensor = preprocess_image(container_img, 512)
                    
                    # Reveal secret
                    with torch.no_grad():
                        revealed_tensor = Rnet(container_tensor).clamp(0, 1)
                    revealed_image = tensor_to_image(revealed_tensor)
                    
                    # Display results
                    st.markdown('<div class="results-container">', unsafe_allow_html=True)
                    st.markdown('<div class="results-title">🎯 Revealing Results</div>', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(container_img, caption="Container Image", use_container_width=True)
                    with col2:
                        st.image(revealed_image, caption="Revealed Secret", use_container_width=True)
                        st.download_button(
                            "📥 Download Revealed Image",
                            data=image_to_bytes(revealed_image),
                            file_name="revealed_secret.png",
                            mime="image/png"
                        )
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.success("✅ Hidden content revealed successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error during revealing: {str(e)}")
        else:
            st.warning("⚠️ Please upload a container image and ensure reveal model is loaded.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 2rem;">
    <p><strong>Neural Style Transfer & Steganography Studio</strong></p>
    <p>Powered by Deep Learning • Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)