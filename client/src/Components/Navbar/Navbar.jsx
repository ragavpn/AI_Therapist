// import logo from '../Images/logo.png'
import { Link } from 'react-router-dom';
import logo from '../../Images/logo.svg'
import loginButton from '../../Images/login_button.svg'
import './Navbar.css'

function Navbar(props) {
    const cur = props.curPage;
    return(
        <div id="navbar" className='flex flex-row justify-between items-center'>
            <img src={logo} alt="" />
            <div id="right-nav">
                <Link to="/home" className={cur === 'landing' ? 'current' : 'not-current'}>Home</Link>
                <Link to="/about" className={cur === 'landing' ? 'current' : 'not-current'}>About</Link>
                <Link to="/contact-us" className={cur === 'landing' ? 'current' : 'not-current'}>Contact Us</Link>
                <Link to="/login">
                    <img src={loginButton} alt="" />
                </Link>
            </div>
        </div>
    )
}

export default Navbar;