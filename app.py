# Main Packages
import streamlit as st

#other packages
from PIL import Image, ImageDraw
import plotly_express as px

#from collection import Counter
from sklearn.cluster import KMeans
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np


##utils 
def load_image(image_file):
    img = Image.open(image_file)
    return img

def get_image_pixel(filename):
    with Image.open(filename) as Rimage:
        rgb_image = np.array(Rimage)
        image_pixel = rgb_image.reshape(-1,3)
        return image_pixel



def create_color_palette(dominant_colors, palette_size=(300, 50)):
    # Create an image to display the colors
    palette = Image.new("RGB", palette_size)
    draw = ImageDraw.Draw(palette)

    # Calculate the width of each color swatch
    swatch_width = palette_size[0] // len(dominant_colors)

    # Draw each color as a rectangle on the palette
    for i, color in enumerate(dominant_colors):
        draw.rectangle([i * swatch_width, 0, (i + 1) * swatch_width, palette_size[1]], fill=tuple(color))

    return palette


def color_analysis(image_pixel,num_clusters):
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters= num_clusters)
    fit = kmeans.fit(image_pixel)
    centroids = kmeans.cluster_centers_
    return centroids

def main():
    st.title("Image Color Extractor")
    menu = ["Home","About"]
    choice = st.sidebar.selectbox("Menu",menu)

    if choice == "Home":
        st.subheader("Home")
        with st.sidebar:
            st.image("src/icons/Img.png")
            st.header("Image Color Extractor")
            st.text("""Image color palette extractor using famous
clustering algorithm K-Means. 
Based on number of colors given,
clusters will be created.""")

        image_file = st.file_uploader("Upload An Image",type = ['PNG','JPG','JPEG'])

        if image_file is not None:
            st.write(":green[Image Upload Complete!]")
            img = load_image(image_file)
            st.image(img)
            
            # analysis 
            #Image pixel
            image_pixel = get_image_pixel(image_file)
            #st.write(image_pixel)

            num_clusters=st.number_input(label='Enter number of colors to extract',step=1,min_value=1,max_value=8)
            if num_clusters is not None:
                if st.button('Select'):
                    pix_df = color_analysis(image_pixel,num_clusters)

                    palette_image = create_color_palette(pix_df.astype(int))

                    st.image(palette_image, caption='Color Palette', use_column_width=True)




    else:
        st.subheader("About")

if __name__ == '__main__':
    main()