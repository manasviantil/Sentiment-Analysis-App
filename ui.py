import streamlit as st
import pickle
import re
import string


# -------------------------
# Page Configuration
# -------------------------

st.set_page_config(
    page_title="Emotion AI Analyzer",
    page_icon="🤖",
    layout="centered"
)


# -------------------------
# Custom CSS
# -------------------------

st.markdown("""
<style>

body{
background-color:#f5f7fb;
}


.title{

font-size:45px;
font-weight:bold;
text-align:center;
color:#4f46e5;

}


.subtitle{

text-align:center;
font-size:18px;
color:#666;

}


.card{

background:white;
padding:30px;
border-radius:20px;
box-shadow:0px 5px 20px rgba(0,0,0,0.1);

}


.result{

text-align:center;
font-size:35px;
font-weight:bold;

}


.stButton button{

width:100%;
height:50px;
font-size:18px;
border-radius:12px;

}


</style>

""",unsafe_allow_html=True)



# -------------------------
# Load Models
# -------------------------

model = pickle.load(
    open("emotion_model.pkl","rb")
)


vectorizer = pickle.load(
    open("tfidf_vectorizer.pkl","rb")
)


emotion_mapping = pickle.load(
    open("emotion_mapping.pkl","rb")
)



# Reverse dictionary

emotion_labels = {
    value:key for key,value in emotion_mapping.items()
}



# -------------------------
# Text Cleaning
# -------------------------

def clean_text(text):

    text=text.lower()


    text=text.translate(
        str.maketrans(
            '',
            '',
            string.punctuation
        )
    )


    text=re.sub(
        r'\d+',
        '',
        text
    )


    text=" ".join(text.split())


    return text



# -------------------------
# Emoji Mapping
# -------------------------

emotion_icons={

"joy":"😊",

"happy":"😄",

"sadness":"😢",

"anger":"😡",

"fear":"😨",

"love":"❤️",

"surprise":"😲",

"disgust":"🤢"

}



# -------------------------
# Header
# -------------------------


st.markdown(
"""
<div class="title">
🤖 Emotion AI Analyzer
</div>

<p class="subtitle">
Understand human emotions using NLP and Machine Learning
</p>

<br>

""",

unsafe_allow_html=True
)



# -------------------------
# Sidebar
# -------------------------


with st.sidebar:

    st.header("About Project")

    st.write(
"""
### NLP Pipeline

✔ Text Cleaning

✔ Stopword Removal

✔ TF-IDF Vectorization

✔ Logistic Regression

✔ Emotion Classification

"""
)


    st.divider()


    st.write(
"""
Developed using:

🐍 Python

🧠 Scikit-Learn

📊 Streamlit

"""
)



# -------------------------
# Main App
# -------------------------


st.markdown(
'<div class="card">',
unsafe_allow_html=True
)


text = st.text_area(

"✍️ Enter your sentence",

placeholder=
"I feel extremely happy today because I got a new job!",

height=150

)



button = st.button(
"🚀 Analyze Emotion"
)



if button:


    if text.strip()=="":


        st.warning(
            "Please enter some text"
        )


    else:


        cleaned = clean_text(text)


        vector = vectorizer.transform(
            [cleaned]
        )


        prediction = model.predict(
            vector
        )[0]


        emotion = emotion_labels[prediction]


        # confidence

        if hasattr(model,"predict_proba"):

            confidence = max(
                model.predict_proba(vector)[0]
            )*100

        else:

            confidence = None



        st.divider()


        icon = emotion_icons.get(
            emotion.lower(),
            "🤖"
        )


        st.markdown(

        f"""

        <div class="result">

        {icon} {emotion.upper()}

        </div>

        """,

        unsafe_allow_html=True

        )


        if confidence:


            st.progress(
                int(confidence)
            )


            st.write(
            f"Confidence : {confidence:.2f}%"
            )



        st.success(
        "Emotion successfully detected!"
        )



st.markdown(
"</div>",
unsafe_allow_html=True
)



st.markdown(
"""
<br>

<center>
Made with ❤️ using NLP & Machine Learning
</center>

""",

unsafe_allow_html=True
)