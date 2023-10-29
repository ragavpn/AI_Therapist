import { useState } from "react"
import { Link } from "react-router-dom"
import './login.css'
import leftVec from "../Images/login_left.png"
import googleLogo from "../Images/googleLogo.svg"
import metaLogo from "../Images/metaLogo.svg"
import appleLogo from "../Images/appleLogo.svg"

function Login() {
    const [formData, setFormData] = useState({
        email: "",
        password: ""
    })

    function handleChange(e) {
        const {name, value} = e.target;
        setFormData(oldData => {
            return ({
                ...oldData,
                [name]: value
            }) 
        })
    }

    function handleSubmit(e) {
        e.preventDefault();
    }

    return (
        <div id="login-page">
            <div id="login-content">
                <img id="login-left" src={leftVec} alt="" />
                <div id="login-right">
                    <p id="checkAcc" style={{color: '#8B7E74'}}>Don't have an account? <span><Link to="/sign-up">Sign Up</Link></span></p>
                    <h1>Log in</h1>
                    <form action="" id="login-form" onSubmit={handleSubmit}>
                        <p>Email <span style={{color: 'red'}}>*</span></p>
                        <input type="email" id="mail-input" name="email" value={formData.email} onChange={handleChange} autoComplete="off" required/>
                        <p>Password <span style={{color: 'red'}}>*</span></p>
                        <input type="password" id="pwd-input" name="password" value={formData.password} onChange={handleChange} required/>
                        <button id="login-button">Log in</button>
                    </form>
                    <p id="or-sign-up-line">or Sign up with</p>
                    <div className="logo-box">
                        <button title="Coming soon" disabled>
                            <img src={googleLogo} alt="" />
                        </button>
                        
                        <button title="Coming soon" disabled>
                            <img src={metaLogo} alt="" />
                        </button>
                        
                        <button title="Coming soon" disabled>
                            <img src={appleLogo} alt="" />
                        </button>
                    </div>
                </div>

            </div>
        </div>
    )
}

export default Login;