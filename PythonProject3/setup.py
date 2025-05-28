from setuptools import setup, find_packages

setup(
    name='music_recommender',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'streamlit',
        'pandas',
        'scikit-learn',
        'joblib',
        'requests',
        'python-dotenv',
    ],
    description='A music recommendation system based on emotion and lyrics similarity',
)