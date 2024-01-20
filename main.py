# %%
import pandas as pd

data = pd.read_csv("passwords.csv")
# %%
data.head()
;# %%
data.rename(columns={"password":"sifre",
            "strength":"guc"}, inplace=True)

data["guc"].replace(1,2, inplace=True)
# %%
data.to_csv("passwords_2.csv", index=False)
# %%
data.info()
# %%
# Veri setinizden şifreleri alın
passwords = list(data["sifre"])

# Her bir şifreyi karakterlere ayırarak tokenleştirme
tokenized_passwords = [[char for char in password] for password in passwords]

# print(tokenized_passwords)

# %%
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.layers import LSTM,Embedding,Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
# %%
tokenizer = Tokenizer()

X = tokenized_passwords
y = data["guc"]
# %%
from sklearn.model_selection import train_test_split
# %%
X_train,X_test,y_train,y_test = train_test_split(X,y, random_state=16)
# %%
X_test
# %%
tokenizer.fit_on_texts(X_train)
# %%
X_train_tr = tokenizer.texts_to_sequences(X_train)
X_test_tr = tokenizer.texts_to_sequences(X_test)
# %%
X_train_pad = pad_sequences(X_train_tr, maxlen=30)
X_test_pad = pad_sequences(X_test_tr, maxlen=30)
# %%
num_words = len(tokenizer.word_index) + 1
# %%
pas_model = Sequential()
# %%
pas_model.add(Embedding(input_dim=num_words,output_dim=50,input_length=30))

pas_model.add(LSTM(75, return_sequences=True))
pas_model.add(LSTM(50, return_sequences=True))
pas_model.add(LSTM(25))

pas_model.add(Dense(1, activation='softmax'))
# %%
pas_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
# %%
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
early_stopper = EarlyStopping(monitor='val_loss', patience=35, verbose=1) 
model_checkpointer = ModelCheckpoint(filepath='password.h5',monitor='val_loss', verbose=0, save_best_only=True)
callback_params = [early_stopper, model_checkpointer]
# %%
pas_model.fit(X_train_pad, y_train, callbacks=callback_params, batch_size=175, epochs=100, validation_split=0.3)
# %%
pas_model.save("password.h5")
# %%
