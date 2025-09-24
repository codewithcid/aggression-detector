import streamlit as st
import librosa
import numpy as np
import os

def analyze_audio_features(audio_file):
    """
    Analyze audio features to detect potential aggression patterns
    This is a demonstration version using acoustic analysis
    """
    try:
        # Load audio
        audio_data, sample_rate = librosa.load(audio_file, sr=22050)
        
        # Extract various audio features
        # 1. MFCC features (voice characteristics)
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1)
        
        # 2. Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sample_rate)[0]
        
        # 3. Zero crossing rate (speech patterns)
        zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
        
        # 4. RMS energy (volume/intensity)
        rms = librosa.feature.rms(y=audio_data)[0]
        
        # Calculate aggregate statistics
        features = {
            'mean_mfcc': np.mean(mfcc_mean),
            'std_mfcc': np.std(mfcc_mean),
            'mean_spectral_centroid': np.mean(spectral_centroids),
            'std_spectral_centroid': np.std(spectral_centroids),
            'mean_spectral_rolloff': np.mean(spectral_rolloff),
            'mean_spectral_bandwidth': np.mean(spectral_bandwidth),
            'mean_zcr': np.mean(zcr),
            'std_zcr': np.std(zcr),
            'mean_rms': np.mean(rms),
            'std_rms': np.std(rms),
            'max_rms': np.max(rms),
            'audio_length': len(audio_data) / sample_rate
        }
        
        return features
    except Exception as e:
        st.error(f"Error analyzing audio: {str(e)}")
        return None

def detect_aggression(features):
    """
    Simple rule-based detection using audio characteristics
    This demonstrates the concept without requiring the problematic model
    """
    if not features:
        return 0.0, "Could not analyze audio"
    
    # Simple heuristic based on audio characteristics
    # Higher values suggest more aggressive patterns
    aggression_score = 0.0
    reasons = []
    
    # High energy/volume patterns
    if features['mean_rms'] > 0.1:
        aggression_score += 0.2
        reasons.append("High average volume detected")
    
    if features['max_rms'] > 0.5:
        aggression_score += 0.15
        reasons.append("Very high peak volume detected")
    
    if features['std_rms'] > 0.15:
        aggression_score += 0.1
        reasons.append("High volume variation detected")
    
    # High frequency content (often associated with stress/anger)
    if features['mean_spectral_centroid'] > 3000:
        aggression_score += 0.15
        reasons.append("High-frequency content detected")
    
    if features['std_spectral_centroid'] > 1000:
        aggression_score += 0.1
        reasons.append("High pitch variation detected")
    
    # Speech pattern irregularities
    if features['std_zcr'] > 0.1:
        aggression_score += 0.1
        reasons.append("Irregular speech patterns detected")
    
    # MFCC-based voice quality indicators
    if features['std_mfcc'] > 15:
        aggression_score += 0.1
        reasons.append("Voice quality variations detected")
    
    # Cap the score at 1.0
    aggression_score = min(aggression_score, 1.0)
    
    return aggression_score, reasons

def main():
    st.title("🎵 Aggressive Behavior Detector (Demo Version)")
    st.write("Upload an audio file for acoustic analysis-based aggression detection")
    
    st.info("""
    **Note**: This is a demonstration version using acoustic feature analysis. 
    It analyzes voice characteristics like volume, pitch, and speech patterns 
    rather than using the original neural network model.
    """)
    
    st.markdown("---")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an audio file", 
        type=['wav', 'mp3', 'ogg'],
        help="Upload a WAV, MP3, or OGG file to analyze"
    )
    
    if uploaded_file is not None:
        # Display file info
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**File:** {uploaded_file.name}")
        with col2:
            st.write(f"**Size:** {uploaded_file.size:,} bytes")
        
        # Audio player
        st.audio(uploaded_file)
        
        # Analysis button
        if st.button("🔍 Analyze Audio Features", type="primary", use_container_width=True):
            with st.spinner("Analyzing acoustic features..."):
                
                # Analyze audio
                features = analyze_audio_features(uploaded_file)
                
                if features:
                    # Detect aggression
                    aggression_score, reasons = detect_aggression(features)
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("🎯 Analysis Results")
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        if aggression_score > 0.5:
                            st.error("⚠️ **POTENTIAL AGGRESSIVE PATTERNS DETECTED**")
                            st.error(f"**Aggression Score:** {aggression_score:.1%}")
                            st.warning("🚨 **ALERT**: Audio shows characteristics often associated with aggressive speech.")
                            
                            if reasons:
                                st.info("**Detection factors:**")
                                for reason in reasons:
                                    st.markdown(f"• {reason}")
                        else:
                            st.success("✅ **CALM SPEECH PATTERNS DETECTED**")
                            st.success(f"**Calm Score:** {(1-aggression_score):.1%}")
                            st.info("😊 Audio shows characteristics of calm, controlled speech.")
                            
                            if reasons:
                                st.info("**Some elevated indicators noted:**")
                                for reason in reasons:
                                    st.markdown(f"• {reason}")
                    
                    with col2:
                        st.markdown("**Score Meter:**")
                        st.progress(aggression_score)
                        st.caption(f"Aggression indicators: {aggression_score:.1%}")
                    
                    # Technical details
                    with st.expander("📊 Detailed Audio Analysis"):
                        st.write("**Audio Features Extracted:**")
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.write(f"Audio Length: {features['audio_length']:.2f}s")
                            st.write(f"Mean Volume (RMS): {features['mean_rms']:.3f}")
                            st.write(f"Peak Volume: {features['max_rms']:.3f}")
                            st.write(f"Volume Variation: {features['std_rms']:.3f}")
                            st.write(f"Mean Pitch Center: {features['mean_spectral_centroid']:.0f} Hz")
                        
                        with col_b:
                            st.write(f"Pitch Variation: {features['std_spectral_centroid']:.0f} Hz")
                            st.write(f"Mean Speech Rate: {features['mean_zcr']:.3f}")
                            st.write(f"Speech Rate Variation: {features['std_zcr']:.3f}")
                            st.write(f"Voice Quality Score: {features['mean_mfcc']:.2f}")
                            st.write(f"Voice Quality Variation: {features['std_mfcc']:.2f}")
                        
                        st.markdown("""
                        **Analysis Method:**
                        - Extracts acoustic features from audio
                        - Analyzes volume, pitch, and speech patterns  
                        - Uses heuristic rules based on research into aggressive speech
                        - Provides indicative results for demonstration
                        """)
                else:
                    st.error("Could not analyze the audio file. Please try a different file.")
    
    else:
        st.info("👆 Upload an audio file to begin analysis")
    
    # Instructions
    st.markdown("---")
    st.markdown("### 📝 How This Demo Works:")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Analysis Process:**
        1. Extracts acoustic features from audio
        2. Measures volume, pitch, and speech patterns
        3. Compares against aggressive speech indicators
        4. Provides percentage-based assessment
        5. Shows detailed feature breakdown
        """)
    
    with col2:
        st.markdown("""
        **Features Analyzed:**
        • Volume levels and variations
        • Pitch characteristics and changes
        • Speech rate and irregularities  
        • Voice quality indicators
        • Spectral content analysis
        """)
    
    st.warning("""
    **Important**: This is a demonstration using acoustic analysis techniques. 
    For production use, a properly trained machine learning model would be more accurate. 
    This version shows how audio features can be used to detect speech patterns associated with aggression.
    """)

if __name__ == "__main__":
    main()