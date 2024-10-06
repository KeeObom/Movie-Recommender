import streamlit as st
import recommender as rec
import base64

# Load the data
movies, ratings = rec.load_data()
user_movie_ratings = rec.create_user_item_matrix(rec.merge_data(movies, ratings))


# Function to get the base and decode the image
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Function to use CSS and set the background to be png_file
def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)




st.title("🎬 Movie Recommendation System 🍿")

# Set background image
set_background('images/background_image_1.jpg')

# Select a user for collaborative filtering
user_id = st.number_input("Enter User ID (1-100):", min_value=1, max_value=100, value=1)

# Select a movie for content-based filtering
movie_title = st.selectbox("Select a movie for content-based recommendations:", movies['title'].unique())

# Display recommendations with explanations
if st.button('Get Recommendations'):
    recommendations, explanations = rec.hybrid_recommendation_with_explanation(user_id, movie_title, user_movie_ratings, movies)
    
    st.write("## 🎥 Recommended Movies 🎥")
    for rec_movie, explanation in zip(recommendations, explanations):
        st.markdown(f"""
        <div style="padding: 10px; border: 2px solid #f1f1f1; border-radius: 8px; margin-bottom: 10px; background-color: #f9f9f9;">
            <h3 style="color: #333;">{rec_movie}</h3>
            
        </div>
        """, unsafe_allow_html=True)
        # st.markdown(f"""
        # <div style="padding: 10px; border: 2px solid #f1f1f1; border-radius: 8px; margin-bottom: 10px; background-color: #f9f9f9;">
        #     <h3 style="color: #333;">{rec_movie}</h3>
        #     <p style="color: #555;">{explanation}</p>
        # </div>
        # """, unsafe_allow_html=True)


# Footer
st.markdown("""
---
Made by [KeeObom](https://github.com/KeeObom)
""")