import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import logo from './logo.svg';
import './App.css';
import Landing from './Landing/landing'
import Login from './Login/login';
import SignUp from './SignUp/signup'


function App() {
  
  return (
    <Router>
        <div className="App">
          <Routes>
            <Route path="/" element={<Landing page="landing"/>} />
            <Route path="/login" element={<Login />} />
            <Route path="/sign-up" element={<SignUp />}/>
          </Routes>
        </div>
    </Router>
      

      
  );
}

export default App;
