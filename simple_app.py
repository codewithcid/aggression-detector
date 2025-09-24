import streamlit as st
import tensorflow as tf
import librosa
import numpy as np
import os

# This function loads our pre-trained AI model
def load_our_model():
    try:
        # Try to load the model (the AI brain)
        if os.path.exists('model.h5'):
            model = tf.keras.models.load_model('model.h5')
            return model
        else:
            st.error("⚠️ Model file not found! Make sure model.h5 is in the same folder as this app.")
            return None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

def prepare_audio_for_ai(audio_file):
    """
    This function prepares the audio file for our AI to understand
    """
    try:
        # Load the audio file
        audio_data, sample_rate = librosa.load(audio_file, sr=22050)
        
        # Extract MFCC features (this is what the AI expects)
        features = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
        features = np.mean(features.T, axis=0)
        
        # Reshape for the model (the AI needs it in this specific format)
        features = features.reshape(1, -1)
        return features
    except Exception as e:
        st.error(f"❌ Error processing audio: {str(e)}")
        return None

# Main app function
def main():
    # App title and description
    st.title("🎵 Aggressive Behavior Detector")
    st.write("Upload an audio file and I'll tell you if it sounds aggressive!")
    st.markdown("---")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an audio file", 
        type=['wav', 'mp3', 'ogg'],
        help="Upload a WAV, MP3, or OGG file to analyze"
    )
    
    # If user uploaded a file
    if uploaded_file is not None:
        # Show file info
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**📁 File:** {uploaded_file.name}")
        with col2:
            st.write(f"**📊 Size:** {uploaded_file.size:,} bytes")
        
        # Let user listen to their file
        st.audio(uploaded_file)
        
        # Analysis button
        if st.button("🔍 Analyze This Audio", type="primary", use_container_width=True):
            with st.spinner("🤔 The AI is listening and analyzing..."):
                
                # Load the AI model
                model = load_our_model()
                
                if model is not None:
                    # Process the audio
                    features = prepare_audio_for_ai(uploaded_file)
                    
                    if features is not None:
                        try:
                            # Ask the AI what it thinks
                            prediction = model.predict(features, verbose=0)
                            confidence = float(prediction[0][0])
                            
                            # Show results
                            st.markdown("---")
                            st.subheader("🎯 Analysis Results")
                            
                            # Create two columns for better layout
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                if confidence > 0.5:
                                    st.error("⚠️ **AGGRESSIVE BEHAVIOR DETECTED!**")
                                    st.error(f"**Confidence Level:** {confidence:.1%}")
                                    st.warning("🚨 **ALERT**: This audio contains patterns associated with aggressive behavior.")
                                    
                                    # Additional alert information
                                    st.info("💡 **Recommended Actions:**")
                                    st.markdown("- Monitor the situation carefully")
                                    st.markdown("- Consider intervention if necessary")
                                    st.markdown("- Document this incident")
                                    
                                else:
                                    st.success("✅ **Normal Speech Detected**")
                                    st.success(f"**Confidence Level:** {(1-confidence):.1%}")
                                    st.info("😊 This audio appears to contain calm, non-aggressive speech patterns.")
                            
                            with col2:
                                # Visual confidence meter
                                st.markdown("**Confidence Meter:**")
                                confidence_to_show = confidence if confidence > 0.5 else (1 - confidence)
                                st.progress(confidence_to_show)
                                
                                if confidence > 0.5:
                                    st.caption(f"🔴 Aggressive: {confidence:.1%}")
                                else:
                                    st.caption(f"🟢 Non-aggressive: {(1-confidence):.1%}")
                            
                            # Technical details
                            with st.expander("📊 Technical Details"):
                                st.write(f"**Raw Prediction Score:** {confidence:.4f}")
                                st.write(f"**Classification Threshold:** 0.5")
                                st.write(f"**Feature Vector Length:** {features.shape[1]}")
                                st.write(f"**Model Type:** Deep Neural Network")
                                
                        except Exception as e:
                            st.error(f"❌ Error making prediction: {str(e)}")
                            st.info("Please try with a different audio file.")
    
    else:
        # Instructions when no file is uploaded
        st.info("👆 Please upload an audio file to get started!")
    
    # Instructions section
    st.markdown("---")
    st.markdown("### 📝 How to Use This App:")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Step-by-step:**
        1. **Upload** an audio file (WAV, MP3, or OGG)
        2. **Listen** to verify it uploaded correctly  
        3. **Click** 'Analyze This Audio' button
        4. **Wait** for the AI to process (takes a few seconds)
        5. **Review** the results and any alerts
        """)
    
    with col2:
        st.markdown("""
        **Tips for best results:**
        - Use clear audio recordings
        - Avoid very short clips (less than 2 seconds)
        - Ensure audio contains speech
        - WAV files usually work best
        - Keep file size under 50MB
        """)
    
    # About section
    with st.expander("🤓 How Does This Work?"):
        st.markdown("""
        **Simple Explanation:**
        This app uses artificial intelligence to analyze speech patterns in audio files. 
        It looks for vocal characteristics that are typically associated with aggressive behavior, 
        such as tone, pitch variations, and speaking patterns.
        
        **Technical Details:**
        - **Model Type:** Deep Neural Network trained on labeled audio data
        - **Input Features:** MFCC (Mel-frequency cepstral coefficients)
        - **Output:** Binary classification (Aggressive vs Non-aggressive)
        - **Training Data:** Various examples of aggressive and calm speech
        - **Accuracy:** Trained to detect patterns with high confidence
        
        **Important Notes:**
        - This is an AI prediction, not a definitive diagnosis
        - Results should be interpreted by qualified professionals
        - Use as a screening tool, not as the sole basis for decisions
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("*Built with Streamlit and TensorFlow | AI-powered behavior analysis*")

if __name__ == "__main__":
    main()