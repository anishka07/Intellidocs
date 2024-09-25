import streamlit as st

from model.intellidocs_main import intelli_docs_main


def main():
    st.title('IntelliDocs RAG Application')

    query = st.text_input('Enter your query:')

    if st.button('Search'):
        if query:
            with st.spinner('Searching...'):
                results = intelli_docs_main(
                    query=query,
                    save_csv_name="streamlit_embeddings.csv",
                    save_csv_dir="CSV_db",
                    rag_device="cpu"
                )

            st.subheader('Search Results:')
            st.write(results)
        else:
            st.warning('Please enter a query.')


if __name__ == '__main__':
    main()
