# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
from pdfplumber import open as pdf_open
import plotly.express as px
from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
import nltk
nltk.download('punkt')

try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_sm")
# Load a pre-trained BERT model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# # Set Streamlit app title and description
# st.title("Document Similarity Comparison")
# st.write("Upload two PDF documents and compare their similarity.")

# Set Streamlit app title and description with styling
st.markdown(
    """
    <div style='background-color: #3399FF; padding: 20px; border-radius: 10px;'>
        <h1 style='color: white; text-align: center;'><b>Document Similarity Analyzer</b></h1>
        <p style='color: white; text-align: center; font-size: 18px;'>
            <i>Upload two PDF documents and explore their similarity.</i>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# Create file upload widgets for two PDF documents
st.sidebar.header("Upload PDF Documents")
document1 = st.sidebar.file_uploader("Upload Document 1", type=["pdf"])
document2 = st.sidebar.file_uploader("Upload Document 2", type=["pdf"])

# Create a function to calculate document similarity
def calculate_similarity(doc1, doc2):
    if doc1 and doc2:
        with pdf_open(doc1) as pdf1, pdf_open(doc2) as pdf2:
            text1 = ""
            text2 = ""
            for page in pdf1.pages:
                text1 += page.extract_text()
            for page in pdf2.pages:
                text2 += page.extract_text()

        # Embed text using the BERT model
        embeddings = model.encode([text1, text2])

        # Compute cosine similarity between the embeddings
        similarity_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

        return similarity_score, text1, text2
    else:
        return None, None, None

def calculate_similarity_pagewise(doc1, doc2):
    if doc1 and doc2:
        with pdf_open(doc1) as pdf1, pdf_open(doc2) as pdf2:
            text1 = [page.extract_text() for page in pdf1.pages]
            text2 = [page.extract_text() for page in pdf2.pages]

        # Embed text for each page using the BERT model
        embeddings1 = model.encode(text1)
        embeddings2 = model.encode(text2)

        # Compute cosine similarity for each page pair
        similarity_matrix = cosine_similarity(embeddings1, embeddings2)

        return similarity_matrix
    else:
        return None

# Create a function to get most frequent words
def get_most_frequent_words(text):
    # Tokenize the text
    words = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
    
    # Count word frequencies
    word_freq = Counter(filtered_words)
    
    # Get the 10 most frequent words
    most_frequent_words = word_freq.most_common(10)
    
    return most_frequent_words

# Create a button to calculate similarity
if st.sidebar.button("Calculate Similarity"):
    # Calculate similarity score
    similarity_score, text1, text2 = calculate_similarity(document1, document2)

    # Create columns
    col1, col2 = st.columns([40,60])

    # Column 1: Most frequent words for Document 1
    with col1:
        if similarity_score is not None:
            # Section 1: Attractive Display of Similarity Score
            st.subheader('Similarity Score')
            similarity_message = ""
            if similarity_score < 0.4:
                similarity_color = "red"
                similarity_message = "Low Similarity"
            elif 0.4 <= similarity_score < 0.7:
                similarity_color = "yellow"
                similarity_message = "Moderate Similarity"
            else:
                similarity_color = "green"
                similarity_message = "High Similarity"
            
            # Large and bold similarity score
            st.markdown(f"<h2 style='color:{similarity_color};'>{similarity_score:.2%}</h2>", unsafe_allow_html=True)
            
            # Similarity level message
            st.markdown(f"<p style='font-size: 18px;'>Similarity Level: <span style='color:{similarity_color}'>{similarity_message}</span></p>", unsafe_allow_html=True)
            
            # Additional information or tips
            st.markdown("<p style='font-size: 14px; color: #555;'>Tip: Upload documents with more text for accurate results.</p>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)
        else:
            # Warning message for no uploaded documents
            st.warning("Please upload two PDF documents first.")

    
    # Most frequent words for Document
    with col2:

        # Section 2: Most Frequent Words
        st.subheader('Most Frequent Words')
        docTab1, docTab2 = st.tabs(["Document 1","Document 2"])

        # Display most frequent words for Document 1
        with docTab1:
            # Card for Document 1
            most_frequent_words_doc1 = get_most_frequent_words(text1)
            # Get all words from Document 2
            all_words_doc1 = [word for word, _ in most_frequent_words_doc1]

            # Combine words into a comma-separated string
            words_string1 = ', '.join(all_words_doc1)

            # Display the words in title case with a larger font size
            st.markdown(f"<h3 style='color: #333; font-size: 20px;'>Frequent Words (Document 1)</h3>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size: 16px;'>{words_string1.title()}</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Display most frequent words for Document 2
        with docTab2:
            # Card for Document 2
            most_frequent_words_doc2 = get_most_frequent_words(text2)
            # Get all words from Document 2
            all_words_doc2 = [word for word, _ in most_frequent_words_doc2]

            # Combine words into a comma-separated string
            words_string = ', '.join(all_words_doc2)

            # Display the words in title case with a larger font size
            st.markdown(f"<h3 style='color: #333; font-size: 20px;'>Frequent Words (Document 2)</h3>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size: 16px;'>{words_string.title()}</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)



    # Section 3: Graphical Representation of Similarity Score using Plotly (Bar Chart)

    similarity_matrix = calculate_similarity_pagewise(document1, document2)
    ############## Visualize the page-wise similarity heatmap ###############
    
    st.subheader('Page-wise Similarity Heatmap')
    max_pages_doc1 = len(similarity_matrix)
    max_pages_doc2 = len(similarity_matrix[0])

    columns = [f'Document 2 - Page {i+1}' for i in range(max_pages_doc2)]
    index = [f'Document 1 - Page {i+1}' for i in range(max_pages_doc1)]

    df_heatmap = pd.DataFrame(similarity_matrix, columns=columns, index=index)

    fig = px.imshow(df_heatmap, labels=dict(x="Document 1", y="Document 2"), color_continuous_scale='Viridis')
    fig.update_layout(title='Page-wise Similarity Heatmap')
    st.plotly_chart(fig)

    ############## Visualize the page-wise similarity line chart ###############
    st.subheader('Page-wise Similarity Line Chart')

    # Extract page numbers for the x-axis
    page_numbers_doc1 = [f'Page {i+1}' for i in range(len(similarity_matrix))]
    page_numbers_doc2 = [f'Page {i+1}' for i in range(len(similarity_matrix[0]))]

    # Flatten the similarity matrix to a 1D array
    similarity_values = similarity_matrix.flatten()

    # Ensure the lengths are consistent
    min_length = min(len(page_numbers_doc1), len(page_numbers_doc2), len(similarity_values))
    page_numbers_doc1 = page_numbers_doc1[:min_length]
    page_numbers_doc2 = page_numbers_doc2[:min_length]
    similarity_values = similarity_values[:min_length]

    # Create a DataFrame with the similarity values and page numbers
    df_similarity = pd.DataFrame({'Page Pair': page_numbers_doc1, 'Similarity Score': similarity_values})

    # Use Plotly Express to generate a line chart
    fig = px.line(df_similarity, x='Page Pair', y='Similarity Score', labels=dict(x="Page Pair", y="Similarity Score"))
    fig.update_layout(title='Page-wise Similarity Line Chart')
    st.plotly_chart(fig)