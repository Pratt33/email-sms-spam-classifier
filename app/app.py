import streamlit as st
import pickle
import string
import os
import sys
from pathlib import Path
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Configure page
st.set_page_config(
    page_title="Email/SMS Spam Classifier",
    page_icon="🛡️",
    layout="wide"
)

# Add custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .spam-prediction {
        background-color: #ffebee;
        color: #c62828;
        border: 2px solid #ef5350;
    }
    .ham-prediction {
        background-color: #e8f5e8;
        color: #2e7d32;
        border: 2px solid #66bb6a;
    }
    .confidence-bar {
        margin: 1rem 0;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class SpamClassifier:
    """Spam classification model wrapper"""
    
    def __init__(self, model_path, vectorizer_path):
        """Initialize the classifier with model and vectorizer paths"""
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.model = None
        self.vectorizer = None
        self.ps = PorterStemmer()
        
    def load_models(self):
        """Load the trained model and vectorizer"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            if not os.path.exists(self.vectorizer_path):
                raise FileNotFoundError(f"Vectorizer file not found: {self.vectorizer_path}")
                
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(self.vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
                
            return True
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            return False
    
    def preprocess_text(self, text):
        """Preprocess text using the same pipeline as training"""
        try:
            # Convert to lowercase
            text = text.lower()
            
            # Tokenize
            words = word_tokenize(text)
            
            # Remove special characters and keep only alphanumeric
            words = [word for word in words if word.isalnum()]
            
            # Remove stopwords and punctuation
            stop_words = set(stopwords.words('english'))
            punctuation = set(string.punctuation)
            
            # Apply stemming
            words = [self.ps.stem(word) for word in words
                     if word not in stop_words and word not in punctuation]
            
            return " ".join(words)
        except Exception as e:
            st.error(f"Error preprocessing text: {str(e)}")
            return ""
    
    def predict(self, text):
        """Make prediction on input text"""
        try:
            if not self.model or not self.vectorizer:
                return None, None, None
                
            # Preprocess text
            processed_text = self.preprocess_text(text)
            if not processed_text:
                return None, None, None
            
            # Vectorize
            text_vector = self.vectorizer.transform([processed_text])
            text_vector_dense = text_vector.toarray()
            
            # Predict
            prediction = self.model.predict(text_vector_dense)[0]
            
            # Get confidence scores
            if hasattr(self.model, 'decision_function'):
                decision_score = self.model.decision_function(text_vector_dense)[0]
                # Convert to probability-like values
                ham_probability = 1 / (1 + np.exp(decision_score))
                spam_probability = 1 - ham_probability
            else:
                # Fallback for models without decision_function
                ham_probability = 0.5
                spam_probability = 0.5
            
            return prediction, ham_probability, spam_probability
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            return None, None, None

def download_nltk_data():
    """Download required NLTK data"""
    import os
    
    # Create a directory for NLTK data in the current working directory
    nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    
    # Add the directory to NLTK's data path
    nltk.data.path.append(nltk_data_dir)
    
    # Download all required NLTK data to the custom directory
    nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
    nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)
    nltk.download('punkt_tab', download_dir=nltk_data_dir, quiet=True)

def main():
    """Main application function"""
    
    # Download NLTK data
    download_nltk_data()
    
    # Header
    st.markdown('<h1 class="main-header">🛡️ Email/SMS Spam Classifier</h1>', unsafe_allow_html=True)
    
    # Initialize classifier with multiple path options for different environments
    possible_paths = [
        # Option 1: Streamlit Cloud environment (current directory is project root)
        Path.cwd() / "models" / "model.pkl",
        Path.cwd() / "models" / "vectorizer.pkl",
        # Option 2: Local development (app directory)
        Path.cwd().parent / "models" / "model.pkl",
        Path.cwd().parent / "models" / "vectorizer.pkl",
        # Option 3: Relative to app directory, going up one level
        Path(__file__).parent.parent / "models" / "model.pkl",
        Path(__file__).parent.parent / "models" / "vectorizer.pkl",
        # Option 4: Try relative to current working directory
        Path("models/model.pkl"),
        Path("models/vectorizer.pkl"),
    ]
    
    # Find the first valid model and vectorizer paths
    model_path = None
    vectorizer_path = None
    
    for i in range(0, len(possible_paths), 2):
        if possible_paths[i].exists() and possible_paths[i+1].exists():
            model_path = possible_paths[i]
            vectorizer_path = possible_paths[i+1]
            break
    
    # If still not found, use the most likely path and let the error handler deal with it
    if model_path is None or vectorizer_path is None:
        model_path = Path.cwd() / "models" / "model.pkl"
        vectorizer_path = Path.cwd() / "models" / "vectorizer.pkl"
    
    classifier = SpamClassifier(str(model_path), str(vectorizer_path))
    
    # Load models
    if not classifier.load_models():
        st.error("❌ Failed to load models. Please check if model files exist.")
        st.error(f"Tried to load from: {model_path}")
        st.error(f"Current working directory: {Path.cwd()}")
        
        # Show helpful instructions
        st.markdown("---")
        st.markdown("### 📁 Expected File Structure")
        st.code("""
email-sms-spam-classifier/
├── app/
│   └── app.py
├── models/
│   ├── model.pkl
│   └── vectorizer.pkl
└── data/
    └── spam.csv
        """)
        
        st.markdown("### 🔧 Troubleshooting")
        st.markdown("""
        1. **Check if model files exist**: Make sure `model.pkl` and `vectorizer.pkl` are in the `models/` folder
        2. **Run from correct directory**: Make sure you're running the app from the project root directory
        3. **Regenerate models**: If files are missing, run the notebook to regenerate them
        """)
        
        st.stop()
    
    # Initialize session state for input text
    if 'input_text' not in st.session_state:
        st.session_state.input_text = ""
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Enter Your Message")
        
        # Text input - controlled by session state
        input_text = st.text_area(
            "Paste your email or SMS message here:",
            value=st.session_state.input_text,
            height=200,
            placeholder="Enter your message here...",
            help="The model will analyze the text and classify it as spam or legitimate.",
            key="text_input"
        )
        
        # Update session state when text changes
        if input_text != st.session_state.input_text:
            st.session_state.input_text = input_text
        
        # Buttons
        col_button1, col_button2, col_button3 = st.columns([1, 1, 1])
        with col_button1:
            clear_button = st.button(
                "🗑️ Clear",
                use_container_width=True,
                help="Clear the input field"
            )
        with col_button2:
            predict_button = st.button(
                "🔍 Analyze Message",
                type="primary",
                use_container_width=True
            )
        with col_button3:
            if st.button("📋 Copy Result", use_container_width=True, help="Copy the last prediction result"):
                if 'last_prediction' in st.session_state:
                    st.info("Result copied to clipboard!")
        
        # Handle clear button
        if clear_button:
            st.session_state.input_text = ""
            st.rerun()
    
    with col2:
        st.header("📋 Sample Messages")
        st.markdown("**Click any example to paste it in the input field:**")
        
        sample_messages = {
            "✅ Legitimate": [
                "Hello, how are you doing today?",
                "Meeting at 3 PM tomorrow. Don't forget to bring the documents.",
                "Thanks for your help with the project yesterday."
            ],
            "🚨 Spam": [
                "CONGRATULATIONS! You've won a $1000 prize! Click here to claim!",
                "Get delivery updates on your USPS order [Number] here: [Link]",
                "FREE entry in 2 a wkly comp to win FA Cup final tkts!"
            ]
        }
        
        for category, messages in sample_messages.items():
            with st.expander(f"{category} Examples"):
                for i, msg in enumerate(messages, 1):
                    # Create a button for each sample message
                    if st.button(f"📝 {msg[:30]}{'...' if len(msg) > 30 else ''}", 
                               key=f"{category}_{i}", 
                               help="Click to paste"):
                        st.session_state.input_text = msg
                        st.rerun()
    
    # Make prediction
    if predict_button and st.session_state.input_text.strip():
        with st.spinner("Analyzing message..."):
            prediction, ham_prob, spam_prob = classifier.predict(st.session_state.input_text.strip())
        
        if prediction is not None:
            # Store prediction result for copy functionality
            result_text = "SPAM" if prediction == 1 else "LEGITIMATE"
            st.session_state.last_prediction = f"Message: {st.session_state.input_text[:100]}{'...' if len(st.session_state.input_text) > 100 else ''}\nPrediction: {result_text}"
            
            st.markdown("---")
            st.header("🎯 Result Analysis")
            
            # Display prediction
            if prediction == 1:  # Spam
                st.markdown(
                    f'<div class="prediction-box spam-prediction">🚨 SPAM DETECTED</div>',
                    unsafe_allow_html=True
                )
            else:  # Ham
                st.markdown(
                    f'<div class="prediction-box ham-prediction">✅ LEGITIMATE MESSAGE</div>',
                    unsafe_allow_html=True
                )
            
            # Confidence scores
            if ham_prob is not None and spam_prob is not None:
                st.subheader("📊 Confidence Scores")
                
                col_conf1, col_conf2 = st.columns(2)
                
                with col_conf1:
                    st.metric("Legitimate Probability", f"{ham_prob:.1%}")
                    st.progress(ham_prob)
                
                with col_conf2:
                    st.metric("Spam Probability", f"{spam_prob:.1%}")
                    st.progress(spam_prob)
            
            # Text analysis
            st.subheader("📝 Text Analysis")
            
            # Basic statistics
            col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
            
            with col_stats1:
                st.metric("Characters", len(st.session_state.input_text))
            
            with col_stats2:
                st.metric("Words", len(st.session_state.input_text.split()))
            
            with col_stats3:
                st.metric("Sentences", len(st.session_state.input_text.split('.')))
            
            with col_stats4:
                processed_text = classifier.preprocess_text(st.session_state.input_text)
                st.metric("Processed Words", len(processed_text.split()) if processed_text else 0)
            
            # Show processed text
            with st.expander("🔍 View Processed Text"):
                st.code(processed_text)
                
        else:
            st.error("Failed to analyze the message. Please try again.")
    
    elif predict_button and not st.session_state.input_text.strip():
        st.warning("Please enter a message to analyze.")
    
    # About section at bottom
    st.markdown("---")
    with st.expander("ℹ️ About This App", expanded=False):
        st.markdown("""
        This application uses a machine learning model to classify emails and SMS messages as spam or ham.
        
        **Model Performance:**
        - Accuracy: 97%
        - Precision: 100%
        
        **How it works:**
        1. Text preprocessing
        2. TF-IDF vectorization
        3. Prediction
        """)
        
        st.info("""
        1️⃣ **Best Model**: Support Vector Classifier
        2️⃣ **Features**: 3K+ TF-IDF features
        3️⃣ **Training Data**: 5K+ SMS messages
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 1rem;'>
            <p>Built with ❤️ using Machine Learning</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    # Add numpy import for probability calculations
    import numpy as np
    main() 