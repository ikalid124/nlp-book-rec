import React, { useState } from "react";
import "./App.css";
import axios from "axios";
import srewashiImg from './srewashi.jpg';
import winnieImg from './winnie.jpg';  

function App() {
  const [prompt, setPrompt] = useState("");
  const [results, setResults] = useState([]);

  const handleSubmit = async () => {
    try {
      const response = await axios.post(`${process.env.REACT_APP_API_URL}/recommend`, {
        prompt,
      });
      setResults(response.data);
    } catch (err) {
      console.error("Error fetching recommendations", err);
    }
  };

  return (
    <div>
    {/* Navbar */}
    <nav className="navbar">
      <div className="nav-logo">📚 ReadMe</div>
      <ul className="nav-links">
        <li><a href="#hero">Home</a></li>
        <li><a href="#team">Our Team</a></li>
      </ul>
    </nav>

    <div>
      {/* Hero Section */}
      <div className="hero">
        <div className="hero-content">
          <div className="hero-icon">📖 ✨</div>
          <h1 className="hero-title">
            Discover Your Next Perfect Read with<span className="hero-highlight"> ReadMe</span>
          </h1>
          <p className="hero-subtitle">
            Tell us what you're in the mood for, and we'll recommend 5 books that will captivate your imagination
          </p>
          <button className="hero-button" onClick={() => window.scrollTo({ top: 600, behavior: "smooth" })}>
            Start Exploring Books
          </button>
        </div>
      </div>
    </div>

      {/* Main Content */}
      <div className="container">
        <h1>
          📚 Rediscover the <span className="highlight">Joy of Reading</span>
        </h1>
        <p>Describe the kind of book you're in the mood for.</p>

        <input
          type="text"
          placeholder="e.g., I want to read books about fantasy with dragons and magic"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
        />
        <button onClick={handleSubmit}>✨ Recommend Books</button>

        <div className="results">
          {results.map((book, index) => (
            <div key={index} className="card">
              <h3 className = 'capitalize'>{book.book_title}</h3>
              <p className = 'capitalize'><strong>Author:</strong> {book.author}</p>
              <p><strong>Genres:</strong> {book.genres}</p>
              <p><strong>Summary:</strong> {book.book_details}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="team-section" id="team">
      <h2>👥 Meet Our Team</h2>
      <p className="team-intro">The passionate people behind ReadMe:</p>

      <div className="team-grid">
        <div className="team-card">
          <img src="https://cosmichomicide.com/wp-content/uploads/2013/09/linkedin-default.png" alt="Kalid Ibrahim" />
          <h3>Kalid I.</h3>
          <p>Role</p>
          <a href="https://www.linkedin.com/in/kalid-ibrahim-b57259355/" target="_blank" rel="noopener noreferrer">
    LinkedIn </a>
        </div>
        <div className="team-card">
          <img src="https://cosmichomicide.com/wp-content/uploads/2013/09/linkedin-default.png" alt="Pranav Satish" />
          <h3>Pranav S.</h3>
          <p>Role</p>
          <a href="https://www.linkedin.com/in/pranav-satish-311a0a250/" target="_blank" rel="noopener noreferrer">
    LinkedIn </a>
        </div>
        <div className="team-card">
          <img src={srewashiImg} alt="Srewashi Mondal" />
          <h3>Srewashi Mondal</h3>
          <p>Role</p>
          <a href="https://www.linkedin.com/in/srewashi-mondal" target="_blank" rel="noopener noreferrer">
    LinkedIn </a>
        </div>
        <div className="team-card">
          <img src={winnieImg} alt="Winnie Tang" />
          <h3>Winnie T.</h3>
          <p>Role</p>
          <a href="https://www.linkedin.com/in/winnie-tang03182/" target="_blank" rel="noopener noreferrer">
    LinkedIn </a>
        </div>
      </div>
    </div>

    <footer className="footer" id="footer">
      <p>© {new Date().getFullYear()} ReadMe. Built by the AI4ALL Team - Section 10 Group B</p>
    </footer>


    </div>
  );
}

export default App;

