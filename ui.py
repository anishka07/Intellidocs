import os

import streamlit as st

from model.intellidocs_main import id_main
from model.llms.gemini_response import gemini_response
from utils.constants import PathSettings


def main():
    st.title("IntelliDocs RAG System")
    st.write("Upload a PDF and query its content.")

    uploaded_pdf = st.file_uploader("Upload your PDF file", type=["pdf"])
    user_query = st.text_input("Enter your query:")
    submit_button = st.button("Submit Query")

    if uploaded_pdf and user_query and submit_button:
        save_pdf_path = os.path.join(PathSettings.PDF_DIR_PATH, uploaded_pdf.name)
        save_csv_name = f"{os.path.splitext(uploaded_pdf.name)[0]}.csv"

        with open(save_pdf_path, "wb") as f:
            f.write(uploaded_pdf.read())

        st.write("Processing your PDF and query...")
        results = id_main(
            save_pdf_name=save_pdf_path,
            user_query=user_query,
            save_csv_name=save_csv_name
        )
        all_response = ''
        for r in results:
            all_response += r['text']

        if results:
            st.write("Here are the relevant results:")
            for result in results:
                st.write(f"**Score:** {result['score']}")
                st.write(f"**Text:** {result['text']}")
                st.write(f"**Page Number:** {result['page_number']}")
                st.write("---")
        else:
            st.write("No relevant results found.")

        llm_response = gemini_response(
            user_query=user_query,
            context=all_response,
        )

        if llm_response:
            st.write("**Here is the result from the LLM:**")
            st.write(llm_response)
        else:
            st.write("No results found.")


if __name__ == "__main__":
    main()
