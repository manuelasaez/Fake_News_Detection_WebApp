from utils import Preprocess, Vectorization
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)  # Import precision_score, recall_score, and f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MaxAbsScaler
from scipy.sparse import issparse
import pickle
import streamlit as st


# Función para cargar el modelo
#@st.cache
def load_model():
    with open('random_forest_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

def main():
# Función principal de la aplicación Streamlit

    st.title("Fake News Text Classification with Random Forest")

    # Cargar el modelo
    model = load_model()

# Subir el archivo CSV
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")

    if uploaded_file is not None:
        # Guardar el archivo temporalmente
        with open("uploaded_file.csv", "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Procesar el archivo CSV
        preprocess = Preprocess()
        _, test_df = preprocess.read_csv("uploaded_file.csv")
        st.write(test_df.head())

        # Procesar datos
        test_df, _ = preprocess.remove_rows()
        test_df, _ = preprocess.remove_duplicates()
        test_df, _ = preprocess.remove_rows_lower_than20()
        test_df, _ = preprocess.newtext()
        test_df = preprocess.filter_english_text_edit_df(test_df, 'new_text')

        # Vectorizar el texto
        object_vectorization = Vectorization()
        filtered_corpus = test_df["new_text"].values
        max_features = 300  # esto se puede cambiar
        unigram_vectors_without_stopwords = object_vectorization.get_tfidf_vectors(
            filtered_corpus, "english", max_features, 1
        )

        # Hacer predicciones
        predictions = model.predict(unigram_vectors_without_stopwords)

        # Mostrar las predicciones
        st.write("Predictions:")
        st.write(predictions)
 # Si el DataFrame tiene una columna de etiquetas verdaderas, evaluamos el rendimiento
        if 'label' in test_df.columns:
            true_labels = test_df['label'].astype(int)  # Convert to integers
            predictions = np.array(predictions).astype(int)  # Convert to integers

            accuracy = accuracy_score(true_labels, predictions)
            st.write(f"Accuracy: {accuracy:.2f}")

    # Mostrar el reporte de clasificación
            report = classification_report(true_labels, predictions, target_names=model.classes_)
            st.text("Classification Report:")
            st.text(report)

    # Mostrar la matriz de confusión
            conf_matrix = confusion_matrix(true_labels, predictions, labels=model.classes_)
            st.write("Confusion Matrix:")
            fig, ax = plt.subplots(figsize=(10, 7))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
            ax.set_xlabel('Predicted labels')
            ax.set_ylabel('True labels')
            st.pyplot(fig)

if __name__ == "__main__":
    main()
