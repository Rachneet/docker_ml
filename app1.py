import streamlit as st
import pickle


file = open("classifier.pkl", "rb")
model = pickle.load(file)

def predict_note_auth(var, skew, kurtosis, entropy):

    prediction = model.predict([[var, skew, kurtosis, entropy]])
    return "The predicted value is : " + str(prediction)

def main():
    st.title("Bank Authenticator")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center">
    Bank Note Authentication App
    </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    var = st.text_input('variance', "Type here")
    skew = st.text_input('skewness', "Type here")
    kurtosis = st.text_input('curtosis', "Type here")
    entropy = st.text_input('entropy', "Type here")
    result = ""
    if st.button("Predict"):
        result = predict_note_auth(var,skew,kurtosis,entropy)
    st.success("The output is: {}".format(result))
    if st.button("About"):
        st.text("This app was built using streamlit")


if __name__== "__main__":
    main()