import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import logo from './logo.svg';
import './App.css';
import Landing from './Landing/landing'
import Login from './Login/login';
import SignUp from './SignUp/signup'
import ChatBot from './ChatBot/chatbot';


function App() {
  
  return (
    <Router>
        <div className="App">
          <Routes>
            <Route path="/" element={<Landing page="landing"/>} />
            <Route path="/login" element={<Login />} />
            <Route path="/sign-up" element={<SignUp />}/>
            <Route path="/home" element={<ChatBot />} />
          </Routes>
        </div>
    </Router>
      

      
  );
}

export default App;
