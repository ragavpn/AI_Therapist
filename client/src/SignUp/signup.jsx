import { useState } from "react"
import { Link } from "react-router-dom"
import './signup.css'
import leftVec from "../Images/login_left.png"
import googleLogo from "../Images/googleLogo.svg"
import metaLogo from "../Images/metaLogo.svg"
import appleLogo from "../Images/appleLogo.svg"
import axios from "axios"

function SignUp() {
    const [formData, setFormData] = useState({
        fullName: "",
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
        console.log(1);
        console.log(formData);
    }

    return (
        <div id="signup-page">
            <div id="signup-content">
                <img id="signup-left" src={leftVec} alt="" />
                <div id="signup-right">
                    <p id="checkAcc" style={{color: '#8B7E74'}}>Have an account? <span><Link to="/login"> Sign Up</Link></span></p>
                    <h1>Sign Up</h1>
                    <form action="" id="signup-form" onSubmit={handleSubmit}>
                        <p>Full name <span style={{color: 'red'}}>*</span></p>
                        <input type="text" id="name-input" name="fullName" value={formData.fullName} onChange={handleChange} autoComplete="off" required/>
                        <p>Email <span style={{color: 'red'}}>*</span></p>
                        <input type="email" id="mail-input" name="email" value={formData.email} onChange={handleChange} autoComplete="off" required/>
                        <p>Password <span style={{color: 'red'}}>*</span></p>
                        <input type="password" id="pwd-input" name="password" value={formData.password} onChange={handleChange} required/>
                    </form>
                        <Link to="/home"><button id="signup-button">Sign up</button></Link>
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

export default SignUp;