import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Modeli yükleme
model = load_model("password.h5")

# Veriyi yükleme
data = pd.read_csv("passwords_2.csv")
max_length = model.input_shape[1]  # Max karakter sayısı, modelin giriş uzunluğu

# Kullanıcıdan şifre girişi
user_password = input("Enter a password: ")

# Modelin kullanabildiği tokenlere dönüştürme
characters = set("".join(data["sifre"]))
char_to_index = {char: index for index, char in enumerate(characters)}

user_tokenized_password = [char_to_index[char] for char in user_password]
user_tokenized_password_padded = pad_sequences([user_tokenized_password], maxlen=max_length, padding='post', truncating='post', dtype='int32')

# Modelle tahmin yapma
user_prediction = model.predict(user_tokenized_password_padded)

# Tahmin sonucunu etiketlere göre belirleme
if user_prediction[0][0] > 0.5:
    predicted_label = "Strong"
    print(f"Entered Password: {user_password}, Predicted Strength: {predicted_label}")
else:
    predicted_label = "Weak"
    print(f"Entered Password: {user_password}, Predicted Strength: {predicted_label}")

