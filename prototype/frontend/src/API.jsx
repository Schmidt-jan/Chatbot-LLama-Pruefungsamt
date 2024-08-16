// api.js

import axios from 'axios';

// Funktion zum Senden der Frage
const sendQuestion = async (question, endpoint = 'lokal') => {
  try {
    console.log(question);

    let url;
    
    // Wähle die URL basierend auf dem Endpoint-Parameter
    if (endpoint === 'openai') {
      url = 'http://localhost:5000/requestsOAI';
    } else {
      url = 'http://localhost:5000/requests';
    }

    // Sende die Anfrage an den ausgewählten Endpoint
    const response = await axios.post(url, {
      question: question
    });

    // Annahme, dass die Antwort die Struktur { answer, pages } hat
    const { answer, pages, link } = response.data;

    // Rückgabe als Objekt, das sowohl answer als auch pages enthält
    return { answer, pages, link };
  } catch (error) {
    console.error('Error:', error);
    throw error;
  }
};



const fetchMessages = async () => {
  try {
    const response = await axios.get('http://localhost:5050/messages');
    return response.data;
  } catch (error) {
    console.error('Error:', error);
    throw error;
  }
};

export { sendQuestion, fetchMessages };
