import React, { Component, useState } from 'react';
import './App.css';
import avatar from './openai-icon.png';
import Logo from './logo_in.svg';
import { sendQuestion } from './API'; // Importiere die Funktionen aus der api.js-Datei
import ChatBot from "react-chatbotify";
import axios from 'axios';


const FooterOpenAI = () => {
  // Zustand zum Speichern der Nachrichten und Seiten
  const [messages, setMessages] = useState([]);
  const [pages, setPages] = useState([]);
  const [link, setLink] = useState([]);

  const formStyle = {
    marginTop: 10,
    marginLeft: 50,
    padding: 10,
    borderRadius: 20,
    maxWidth: 300,
    backgroundColor: '#491D8D', // Matching the chat bubble's background color
    color: 'white', // Text color inside the bubble
    border: 'none', // Remove border to match the bubble
    fontSize: '15px', // Adjust font size to match the chat bubble
    boxShadow: '0 2px 10px rgba(0, 0, 0, 0.1)', // Optional: Adds a shadow for a more realistic bubble look
  };
  
  // Funktion zum Verarbeiten der Frage und Rückgabe der Antwort und Seiten
  const sendQuestionHandler = async (question) => {
    try {
      const { answer, pages, link } = await sendQuestion(question, 'openai');
      console.log('Answer:', answer);
      console.log('Pages:', pages);
      console.log('Link:', link);
      // Aktualisiere den Zustand mit der neuen Nachricht und der Antwort des Bots
      setMessages(prevMessages => [
        ...prevMessages,
        { text: question, user: 'user' },
        { text: answer, user: 'bot' }
      ]);
      setPages(pages); // Setze pages direkt
      setLink(link);

      // Rückgabe von Antwort und Seiten für die Verwendung im Flow
      return { answer, pages, link };
    } catch (error) {
      console.error('Error:', error);
      return { answer: 'Es gab einen Fehler bei der Verarbeitung Ihrer Anfrage.', pages: [], link: link };
    }
  };


  const stream = async (params) => {
		try {
			
			const response = await axios.get(`http://localhost:8000/requestsOAI?question=${params.userInput}`,{
        adapter: 'fetch',
        responseType: 'stream'
      }).then(async (response) => {
        const stream = response.data;
        //const completeResponse = await new Response(stream).text()
        let text = "";
			  let offset = 0;
        // consume response
        const reader = stream.pipeThrough(new TextDecoderStream()).getReader();

        await reader.read().then(async function pump({ done, value }) {
          if (done) {
            return;
          }
          text += value
          //  await params.streamMessage(text) -> more chunky but faster generation
          for (let i = offset; i < text.length; i++) {
            // stream message character by character
            await params.streamMessage(text.slice(0, i + 1));
            //await new Promise(resolve => setTimeout(resolve, 30)); -> slows down generation and makes it less chunky
          }
          offset += value.length


          return reader.read().then(pump);
        });
      })
		} catch (error) {
      console.error(error)
			await params.injectMessage("Error while retrieving response");

		}

    try {
      const response = await axios.get('http://localhost:8000/pages')
      console.log(response)
      const {pages, link} = response.data
      setPages(pages[0]); // Setze pages direkt
      setLink(link);
    } catch (error) {
      console.error(error)
			await params.injectMessage("Error while retrieving document sources");

		}
	}






  // Definiere den Flow des Chatbots
  const flow = {
    start: {
      message: "Hi! Wie kann ich dir helfen?",
      path: "askQuestion"
    },
    askQuestion: {
      message: async (params) => {
        /*
        const userInput = params.userInput;
        console.log('User input:', userInput);
        // Frage senden und auf die Antwort warten
        const { answer, pages, link } = await sendQuestionHandler(userInput);
        console.log('BP1', answer, pages, link);
        // Antwort und Seiten für den nächsten Schritt zurückgeben
        return answer;
        */
        await stream(params)
      },
      path: "giveSource", // Weiterleitung zum nächsten Schritt
      transition: { duration: 1000 }
    },
    giveSource: {
      message: "",
      component: (
        <div style={formStyle}>
          <p>Die genauen Angaben findest du auf dieser/diesen Seite(n): 
          {
            (() => {
                if (!pages || pages?.length === 0) {
                      return (
                        <a href={link} style={{ color: 'white' }} target="_blank" rel="noopener noreferrer"> hier</a>
                      )
                } else {
                    return (
                      pages.map((page) => {
                        return <span> <a href={link + '#page=' + page} style={{ color: 'white' }} target="_blank" rel="noopener noreferrer">{page}</a>,</span>
                      })
                    )
                }
            })()  
          }
          </p>
        </div>
      ),
      path: "askNextQuestion", // Weiterleitung zum nächsten Schritt
      transition: { duration: 2000 }
    }
    ,
    askNextQuestion: {
      message: "Hast du noch weitere Fragen?",
      path: "askQuestion", // Rückkehr zur Frage-Stufe
    }
  };

  // Konfiguration für den ChatBot
  const styles = {
    headerStyle: {
      background: '#ffffff',
      color: '#ffffff',
      padding: '10px',
    },
    chatWindowStyle: {
      backgroundColor: '#D9E5EC',
    },
    // ...andere Stile
  };

  const config = {
    chatHistory: { disabled: true },
    botBubble: {
      avatar: avatar,
      showAvatar: true,
      simStream: true,
      streamSpeed: 20,
      dangerouslySetInnerHtml: true
    },
    emoji: { disabled: true },
    fileAttachment: { disabled: true },
    footer: { text: 'Bitte beachte, dass ich nur eine KI bin und Fehler machen kann, kontrolliere daher alle Aussagen!' },
    header: { avatar: Logo, showAvatar: true },
    notification: { disabled: true, showCount: false },
    tooltip: { text: 'Fragen? Vielleicht kann ich weiterhelfen!', avatar: avatar },
    chatButton: { icon: avatar },
    chatInput: { enabledPlaceholderText: 'Tippe deine Frage hier ein...' },
  };

  return (
    <div className="footer">
      <ChatBot settings={config} styles={styles} flow={flow} />
    </div>
  );
}

export default FooterOpenAI;
