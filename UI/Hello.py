import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋"
)

st.write("# Welcome to my Home Page! 👋")

st.sidebar.success("Select an option to match your own desire.")

st.markdown(
    """
    This is a full-stack online scene reconstruction software, which includes deep network training, deep graph
    inference, and point cloud visualization functions. The software performance metrics achieve an average pixel
    error (EPE) for stereo disparity estimation of less than 0.8. For images with a resolution of 960*540,
    the frame rate for stereo disparity estimation is not less than 15 frames per second (FPS).

    #### **👈 Select a demo from the sidebar** to see some examples of what Streamlit can do!

    ### Want to learn more?
    - Check out [streamlit.io](https://streamlit.io)
    - Jump into our [documentation](https://docs.streamlit.io)
    - Jump into our repositories in github[github](https://github.com/jason666ke/ZED)

    ### Technical Details of each Feature
    - Disparity Calculation: Provides two methods for calculation—traditional SGBM algorithm
    and deep calculation using [FADNet](https://github.com/HKBU-HPML/FADNet-PP).
    - Depth Calculation: Based on the formula $depth = \\frac{fx * baseline}{disparity}$, where:
        - $fx$ represents the focal length,
        - $Baseline$ refers to the distance between the two cameras,
        - $Disparity$ indicates the difference in image location between the same object points seen by the two cameras.
    - Point Cloud Calculation and Visualization:
    Utilizes the [Open3D library](https://www.open3d.org/docs/release/index.html)
    for both point cloud calculation and visualization.
    """
)




