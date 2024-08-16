import React, { useState } from 'react';
import './App.css';
import Footer from './Footer'; // Importiere die Footer-Komponente
import FooterOpenAI from './FooterOpenAI'; // Importiere die FooterOpenAi-Komponente
import logo from './logo_in.svg';
import login from './login.png';
import Toggle from "react-toggle";
import OAIAvatar from './LogoOpenAI.jpg';
import LlmAvatar from './avatar.png';

const Header = () => {
  return (
    <div className="header">
      <img src={logo} alt="Logo" className="logo" />
      <div className="tabs">
        {/* Hier könnten Tabs hinzugefügt werden */}
        <span>Studium</span>
        <span>Hochschule</span>
        <span>Forschung und Transfer</span>
        <img src={login} alt="Login" className="login-icon" />
      </div>
      <h1>Prüfungsangelegenheiten</h1>
      <p>Auf der Seite Prüfungsangelegenheiten finden Sie alle Studien- und Prüfungsordnungen (SPO), 
        Zulassungssatzungen, Modalitäten zur Prüfungsan- und -abmeldung, Notenberechnung und vieles mehr.</p>
    </div>
  );
};

const MainContent = () => {
  return (
    <div className="main-content">
      {/* Beispielabschnitte */}
      <section>
        <h2>Während Prüfungszeitraum sind nur manche Noten sichtbar!</h2>
        <p>Jedes Semester (während des Prüfungszeitraums wenn Noten online verbucht werden) ist der Notenspiegel nur teilweise aktuell (WS: ca. zweite Januarwoche bis erste Märzwoche; SS: Anfang Juli bis ca. 5. August). <br /> <br />

Die Noten des aktuellen Semesters, also die jeweiligen Ergebnisse Ihrer aktuell erbrachten Prüfungsleistungen, werden während des Prüfungszeitraums online durch die Prüferinnen und Prüfer und im Zentralen Prüfungsamt eingetragen.  <br />
Wenn Sie sich also während des Prüfungszeitraums Ihren Notenspiegel ausdrucken, fehlen die aktuellen Noten und auch die Datensätze, die evtl. aus früheren Semestern bereits erfasst sind, aber noch nicht in die Modulnote integriert sind. <br /><br />

Dies führt regelmäßig zu Rückfragen, etwa so: "Warum sehe ich nur manche Noten?" oder "Gestern war die Prüfung noch im Anmeldestatus!"</p>
      </section>
      <section>
        <h2>Abschnitt 2</h2>
        <p>Von Ihrem Studiengang erfahren Sie die jeweiligen Termine, wann die Noten des aktuellen Semesters sichtbar sind. Grundsätzlich kann gesagt werden, dass ca. 8-10 Tage NACH DEM ENDE DES PRÜFUNGSZEITRAUMS wieder alle Noten sichtbar sind, also auch die aus früheren Semestern und Sie können Ihren vollständigen Notenspiegel über das Portal ausdrucken.</p>
      </section>
      {/* ... weitere Abschnitte hier ... */}
    </div>
  );
};

  function App() {
    // Zustand für den Toggle-Button
    const [showFooterOpenAI, setShowFooterOpenAI] = useState(false);
  
    // Toggle-Funktion
    const toggleFooter = () => {
      setShowFooterOpenAI(!showFooterOpenAI);
    };
  
    return (
      <div className="App">
        <div className="upper-quarter">
          <Header />
        </div>
        <div className="lower-three-quarters">
          <MainContent />
          <div className="footer-container">
            {/* Toggle Button */}
            <button onClick={toggleFooter} className="toggle-button">
              {showFooterOpenAI ? 'Lokales LLM verwenden' : 'OpenAI-API verwenden'}
            </button>
            {/* Bedingte Anzeige der Fußzeile */}
            {showFooterOpenAI ? <FooterOpenAI /> : <Footer />}
          </div>
        </div>
      </div>
    );
  }
  
  export default App;