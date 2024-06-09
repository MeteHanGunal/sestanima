import sys
import pickle
import threading
import speech_recognition as sr
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QTextEdit, QLabel
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt

class SpeechTopicClassifierApp(QWidget):
    wordCountChanged = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.initUI()
        self.loadModel()
        self.listening = False
        self.audio_data = []
        self.word_count = 0

        # Sinyal ve slot bağlantısı
        self.wordCountChanged.connect(self.updateWordCountLabel)

    def initUI(self):
        self.setWindowTitle("Konuşma Konusu Sınıflandırıcı")
        self.setGeometry(100, 100, 600, 400)

        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("Konuşma Sonuçları Burada Gösterilecek")
        self.text_edit.setReadOnly(True)

        self.start_button = QPushButton("Başla")
        self.start_button.clicked.connect(self.startListening)

        self.stop_button = QPushButton("Dur")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stopListening)

        self.word_count_label = QLabel("Kelimelerin Sayısı: 0")

        vbox = QVBoxLayout()
        vbox.addWidget(self.text_edit)
        vbox.addWidget(self.start_button)
        vbox.addWidget(self.stop_button)
        vbox.addWidget(self.word_count_label)

        self.setLayout(vbox)

        self.recognizer = sr.Recognizer()

    def loadModel(self):
        with open('turkish_text_classification_model.pkl', 'rb') as model_file:
            self.model = pickle.load(model_file)
        print("Model başarıyla yüklendi.")

    def startListening(self):
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.listening = True
        self.audio_data = [] 
        self.word_count = 0
        self.wordCountChanged.emit(self.word_count)
        self.listenSpeech()

    def stopListening(self):
        self.listening = False
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.processAudio() 

    def listenSpeech(self):
        def recognize():
            with sr.Microphone() as source:
                print("Konuşmayı Dinle...")
                self.recognizer.adjust_for_ambient_noise(source)
                while self.listening:
                    print("Dinleme işlemi sürüyor...")
                    try:
                        audio = self.recognizer.listen(source)
                        self.audio_data.append(audio)
                    except sr.WaitTimeoutError:
                        print("Zaman aşımı, tekrar dinleniyor...")
        
        recognition_thread = threading.Thread(target=recognize)
        recognition_thread.start()

    def processAudio(self):
        print("Ses tanıma ve sınıflandırma işlemi başlatılıyor...")
        try:
            combined_text = ""
            for audio in self.audio_data:
                text = self.recognizer.recognize_google(audio, language="tr-TR")
                combined_text += " " + text
                self.word_count += len(text.split())
            self.wordCountChanged.emit(self.word_count)
            topic = self.classifyTopic(combined_text)
            self.text_edit.append(f"Ses Tanındı: {combined_text}")
            self.text_edit.append(f"Konuşmanın Konusu: {topic}")
        except sr.UnknownValueError:
            print("Üzgünüm, ses anlaşılamadı.")
            self.text_edit.append("Üzgünüm, ses anlaşılamadı.")
        except sr.RequestError as e:
            print(f"Ses tanıma servisine ulaşılamadı; {e}")
            self.text_edit.append(f"Ses tanıma servisine ulaşılamadı; {e}")

    def classifyTopic(self, text):
        predicted_label = self.model.predict([text])[0]
        return predicted_label

    @pyqtSlot(int)
    def updateWordCountLabel(self, count):
        self.word_count_label.setText(f"Kelimelerin Sayısı: {count}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = SpeechTopicClassifierApp()
    ex.show()
    sys.exit(app.exec_())
