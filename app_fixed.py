import streamlit as st
import tensorflow as tf
import librosa
import numpy as np
import os
import json

# Configure TensorFlow to avoid warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_our_model():
    """
    Try multiple methods to load the model
    """
    try:
        # Method 1: Try loading from JSON + weights (most reliable for old models)
        if os.path.exists('model.json') and os.path.exists('model.h5'):
            st.info("Loading model from JSON + weights...")
            with open('model.json', 'r') as json_file:
                model_json = json_file.read()
            model = tf.keras.models.model_from_json(model_json)
            model.load_weights('model.h5')
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            return model
    except Exception as e:
        st.warning(f"Method 1 failed: {str(e)}")
    
    try:
        # Method 2: Try loading the complete model without compilation
        if os.path.exists('model.h5'):
            st.info("Trying to load complete model...")
            model = tf.keras.models.load_model('model.h5', compile=False)
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            return model
    except Exception as e:
        st.warning(f"Method 2 failed: {str(e)}")
    
    try:
        # Method 3: Force load with custom objects
        if os.path.exists('model.h5'):
            st.info("Trying force load...")
            model = tf.keras.models.load_model('model.h5', compile=False, custom_objects={})
            return model
    except Exception as e:
        st.error(f"All loading methods failed: {str(e)}")
        st.error("The model file might be incompatible with the current TensorFlow version.")
        return None

def prepare_audio_for_ai(audio_file):
    """
    Process audio file for the AI model
    """
    try:
        # Load the audio file
        audio_data, sample_rate = librosa.load(audio_file, sr=22050)
        
        # Extract MFCC features
        features = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
        features = np.mean(features.T, axis=0)
        
        # Ensure we have the right shape
        features = features.reshape(1, -1)
        return features
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return None

def main():
    st.title("🎵 Aggressive Behavior Detector")
    st.write("Upload an audio file and I'll tell you if it sounds aggressive!")
    st.markdown("---")
    
    # Check if model files exist
    if not (os.path.exists('model.h5') and os.path.exists('model.json')):
        st.error("Model files not found! Make sure both model.h5 and model.json are in the same folder.")
        st.stop()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an audio file", 
        type=['wav', 'mp3', 'ogg'],
        help="Upload a WAV, MP3, or OGG file to analyze"
    )
    
    if uploaded_file is not None:
        # Show file info
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**File:** {uploaded_file.name}")
        with col2:
            st.write(f"**Size:** {uploaded_file.size:,} bytes")
        
        # Audio player
        st.audio(uploaded_file)
        
        # Analysis button
        if st.button("🔍 Analyze This Audio", type="primary", use_container_width=True):
            with st.spinner("Loading AI model and analyzing audio..."):
                
                # Load model
                model = load_our_model()
                
                if model is not None:
                    st.success("✅ Model loaded successfully!")
                    
                    # Process audio
                    features = prepare_audio_for_ai(uploaded_file)
                    
                    if features is not None:
                        try:
                            # Make prediction
                            with st.spinner("AI is analyzing the audio patterns..."):
                                prediction = model.predict(features, verbose=0)
                                confidence = float(prediction[0][0])
                            
                            # Show results
                            st.markdown("---")
                            st.subheader("🎯 Analysis Results")
                            
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                if confidence > 0.5:
                                    st.error("⚠️ **AGGRESSIVE BEHAVIOR DETECTED!**")
                                    st.error(f"**Confidence Level:** {confidence:.1%}")
                                    st.warning("🚨 **ALERT**: This audio contains patterns associated with aggressive behavior.")
                                    
                                    st.info("**Recommended Actions:**")
                                    st.markdown("• Monitor the situation carefully")
                                    st.markdown("• Consider appropriate intervention")
                                    st.markdown("• Document this incident")
                                    
                                else:
                                    st.success("✅ **Normal Speech Detected**")
                                    st.success(f"**Confidence Level:** {(1-confidence):.1%}")
                                    st.info("😊 This audio appears to contain calm, non-aggressive speech.")
                            
                            with col2:
                                st.markdown("**Confidence Meter:**")
                                confidence_to_show = confidence if confidence > 0.5 else (1 - confidence)
                                st.progress(confidence_to_show)
                                
                                if confidence > 0.5:
                                    st.caption(f"🔴 Aggressive: {confidence:.1%}")
                                else:
                                    st.caption(f"🟢 Non-aggressive: {(1-confidence):.1%}")
                            
                            # Technical details
                            with st.expander("📊 Technical Details"):
                                st.write(f"**Raw Score:** {confidence:.4f}")
                                st.write(f"**Threshold:** 0.5")
                                st.write(f"**Features:** {features.shape[1]} MFCC coefficients")
                                st.write(f"**Model:** Deep Neural Network")
                                
                        except Exception as e:
                            st.error(f"Error during prediction: {str(e)}")
                else:
                    st.error("❌ Could not load the AI model. Please check the model files.")
    
    else:
        st.info("👆 Upload an audio file to get started!")
    
    # Instructions
    st.markdown("---")
    st.markdown("### 📝 How to Use:")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Steps:**
        1. Upload an audio file
        2. Listen to verify upload
        3. Click 'Analyze This Audio'  
        4. Wait for AI processing
        5. Review results and alerts
        """)
    
    with col2:
        st.markdown("""
        **Tips:**
        • Use clear recordings
        • Avoid very short clips
        • WAV files work best
        • Keep under 200MB
        • Ensure speech content
        """)
    
    with st.expander("🤓 How This Works"):
        st.markdown("""
        This app uses deep learning to analyze speech patterns in audio files. 
        It extracts acoustic features and compares them to patterns learned 
        from training data containing aggressive and non-aggressive speech examples.
        
        **Technical:** Uses MFCC features and a neural network trained on labeled audio data.
        **Note:** This is an AI screening tool, not a definitive diagnosis.
        """)

if __name__ == "__main__":
    main()