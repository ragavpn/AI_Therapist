// import ChatNavBar from "../Components/ChatNav/Navbar"
import socketIOClient from "socket.io-client"
import './chatbot.css'
import { useReactMediaRecorder } from 'react-media-recorder'
import axios from 'axios'
// import menu from "../Images/menu.png"
import menu from "../Images/hammenu.svg"
import settings from "../Images/settings.svg"
import userPP from "../Images/userProfile.svg"
import plus from "../Images/plus.svg"
import fileIcon from "../Images/fileIcon.svg"
import arrow from "../Images/arrow.svg"
import leftArrow from "../Images/left_arrow.svg"
import rightArrow from "../Images/right_arrow.svg"
import mic from "../Images/mic.svg"
import deliver from "../Images/deliver.svg"
import squiggle from  "../Images/squiggle.svg"
import '../Components/ChatNav/Navbar.css'

import { useState, useEffect } from 'react'



function ChatBot() {

    // const { status, startRecording, stopRecording, mediaBlobUrl} = useReactMediaRecorder({ audio: true});
    
    const [input, setInput] = useState({
        query: "",
    })
    const formData = {}

    const [userMessages, setUserMessages] = useState([]);
    const [botMessages, setBotMessages] = useState([]);
    const [messages, setMessages] = useState([]);
    const [userQueries, setUserQueries] = useState([]);
    const [botQuery, setBotQueries] = useState([]);
    const socket = socketIOClient('http://localhost:5000'); 

    // useEffect(() => {
    //     socket.on('message', (message) => {
    //         setMessages((prevMessages) => [...prevMessages, message]);
    //     });

    //     return () => {
    //         // Clean up when the component unmounts
    //         socket.disconnect();
    //     };
    // }, [])

    const handleSendMessage = () => {
        // Send a message to the server
        const newMessage = {
            text: input.query,
            user: 'user', // Set the sender as 'user'
        };
        setMessages((prevMessages) => [...prevMessages, newMessage]);

        axios.get("http://localhost:5000/message", {
            params: {
                userInput: newMessage.text
            }
        })
            .then(res => {
                setMessages((prevMessages) => [...prevMessages, {text: res.data, user: 'bot'}]);
            })
        // setUserMessages((prevMessages) => [...prevMessages, newMessage]);
        // Emit a 'message' event to the server
        console.log(messages)
        // socket.emit('message', newMessage);

        // Update the local state to display the sent message

        // Clear the input field
        setInput({ query: '' });
    };

    socket.on('message', (message) => {
        setMessages([...messages, { text: message, user: 'bot' }]);
    });


    function handleChange(e) {
        const {name, value} = e.target;
        setInput(old => {
            return ({
                ...old, 
                [name]: value,
            })
        })
    }

    // function handleSubmit() {
        
    //     // formData['audio'] = mediaBlobUrl;
    //     formData.language = languageMap[langNum];
    //     formData['textData'] = input.query;
    //     setInput({query: ""})
    //     console.log(formData)
    //     axios.post('http://localhost:5000/api/get-audio', formData)
    //       .then(res => console.log(res.data))
    // }

    const [isRecording, setIsRecording] = useState(false);
    const [langNum, setLangNum] = useState(1);
    // const [curLang, setCurLang] = useState("English")
    const [numChats, setNumChats] = useState([1, 1, 1]);
    const [showAll, setShowAll] = useState(false);

    const [feed, setFeed] = useState(0)
    
    const [showSideBar, setShowSideBar] = useState(true);

    const languageMap = {
        1: 'English',
        2: 'Hindi',
        3: 'Tamil'
    }

    const sideBarStyle = {
        transform: `translateX(${!showSideBar ? '-130%' : '0%'})`,
        paddingTop: '2%',
        paddingLeft: '3%',
        backgroundColor: '#8B7E74',
        width: '19.75rem',
        height: '35.25rem',
        borderRadius: '20px',
        color: '#F1D3B3',
        textAlign: 'left',
        transition: 'all 200ms',
    }
    const inpMetaStyles = {
        // transform: `translateX(${!showSideBar ? '-130%' : '0%'})`,
        transform: `translateX(${!showSideBar ? '-10%' : '0%'})`,
        display: 'flex',
        // backgroundColor: 'red',
        justifyContent: 'space-beteen',
        alignItems: 'center',
        marginTop: '3%',
        paddingLeft: '2%',
        transition: 'all 200ms',
    }
    const contentStyles = {
        transform: `translateX(${!showSideBar ? '-2.5%' : '0%'})`,
        marginTop: '2.5%',
        // backgroundColor: 'aqua',
        transition: 'all 200ms'
    }
    
    const inputLeftStyles = {
        transform: `translateX(${!showSideBar ? '-150%' : '0%'})`,
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
    }
    const arrowStyles = {
        transform: `rotate(${showAll ? '180deg' : '0deg'}) translateY(-20%)`,
        // transform: 'translateY(-20%)',
        marginRight: '10%',
        transition: 'all 200ms'
    }

    const chatDisplayStyles = {
        transform: `translateX(${!showSideBar ? '-10%' : '0%'})`,
        marginLeft: '9%',
        marginRight: '5%',
        backgroundColor: 'black',
        borderRadius: '30px',
        height: '565px',
        width: '100%'

    }

    function addChat() {
        setNumChats(oldChat => {
            return(
                [...oldChat, 1]
            )
        })
    }

    function toggleChats() {
        setShowAll(old => {
            console.log(old);
            return !old;
        })
    }

    const defaultChatElements = () => {
        for(let i=0; i<3; i++) {
            return (
                <div className='cur-chats'>
                    <img src={fileIcon} alt="" />
                    <p>Lorem impus</p>
                </div>
            )
        }
    }

    const allChatElements = numChats.map(chat => {
        return (
            <div className='cur-chats'>
                <img src={fileIcon} alt="" />
                <p style={{marginLeft: '10%'}}>Lorem impus</p>
            </div>
        )
    })

    function prevLang() {
        if(langNum > 1) {
            setLangNum(old => {
                return old-1;
            })
            console.log(langNum)
            // setLang();
            // setCurLang(languageMap[langNum])
        }
    }
    function nextLang() {
        if(langNum < 3) {
            setLangNum(old => {
                return old+1;
            })
            // console.log(langNum)
            
            // setCurLang(languageMap[langNum])
        }
    }


    
    function toggleSideBar() {
        setShowSideBar(old => !old);
    }

    function send1() {
        // setFeed(1);
        // socket.emit('feedback', 1)
        axios.post("http://localhost:5000/feedback", {
            params: {
                value: 1
            }
        })
    }
    function send2() {
        // setFeed(2);
        // socket.emit('feedback', 2)
        axios.post("http://localhost:5000/feedback", {
            params: {
                value: 2
            }
        })
    }
    function send3() {
        // setFeed(3);
        // socket.emit('feedback', 3)
        axios.post("http://localhost:5000/feedback", {
            params: {
                value: 3
            }
        })
    }
    function send4() {
        // setFeed(4);
        // socket.emit('feedback', 4)
        axios.post("http://localhost:5000/feedback", {
            params: {
                value: 4
            }
        })
    }
    function send5() {
        // setFeed(5);
        // socket.emit('feedback', 5)
        axios.post("http://localhost:5000/feedback", {
            params: {
                value: 5
            }
        })
    }


    const chatMsgs = messages.map(msg => {
        return (
            <div className={`${msg.user === 'user' ? 'user-message' : 'bot-message'}`}>{ msg.user}: {msg.text}</div>
        )
    })
    

    return(
        <div id="chatbot-page">
            <div id="chat-nav-bar">
                <img id="ham-menu" onClick={toggleSideBar} src={menu} alt="" />
                <div id="chat-nav-right">
                    <img src={settings} alt="" />
                    <img id="user-profile-pic" src={userPP} alt="" />
                </div>
            </div>
            <div id="content" style={contentStyles}>
                <div id="sidebar" style={sideBarStyle}>
                    <button id="new-chat-button" onClick={addChat}> 
                        <div id="new-button">
                            <img src={plus} />
                            <p>New chat</p>
                        </div>
                    </button>
                    <div id="chats">
                        <p>Recent</p>
                        <div id="cur-chat-display">
                            {/* {showAll ? {allChatElements} : defaultChatElements} */}
                            {allChatElements}
                        </div>
                        <button onClick={toggleChats}>
                            <div id="show-more-button-box">
                                <img id="arrow" style={arrowStyles} src={arrow} alt="" />
                                <p style={{color: '#5A4739', display:'inline-block'}}>Show more</p>
                            </div>
                        </button>
                    </div>
                </div>
                <main id="chat-display" style={chatDisplayStyles}>
                    {chatMsgs}
                </main>
                
            </div>

            <div id="input-meta" style={inpMetaStyles}>
                <div id="input-left" style={inputLeftStyles}>
                    <div id="or-talk">Or Talk to us</div>
                    <div id="lang-select">
                        
                        <button onClick={prevLang} id="left-button" className={langNum === 1 ? `disabled` : `enabled`}><img src={leftArrow} alt="" /></button>
                        <p>{ languageMap[langNum] }</p>                        
                        <button onClick={nextLang} id="right-button" className={langNum === 3 ? `disabled` : `enabled`}><img src={rightArrow} alt="" /></button>
                        
                    </div>
                    <img id="squiggle" src={squiggle} alt="" />
                </div>
                <div id="input-right">
                    <div id="modes">
                        <div id="feedback">
                            <p>Rate the response- </p>
                            <div id="emojis">
                                <div onClick={send1}>😖</div>
                                <div onClick={send2}>🙁</div>
                                <div onClick={send3}>😔</div>
                                <div onClick={send4}>🙂</div>
                                <div onClick={send5}>😍</div>
                            </div>
                        </div>
                    </div>
                    <input id="user-input" name="query" value={input.query} placeholder="Share Your Thoughts..." type="text" onChange={handleChange}/>
                        <img id="voice-inp" src={mic} alt=""/>
                        <img id="send-prompt" onClick={handleSendMessage} src={deliver} alt="" />

                        {/* {
                            mediaBlobUrl && <audio controls id="user-audio">
                                <source src={mediaBlobUrl}/>
                            </audio>
                        } */}
                    {/* </input> */}
                </div>
            </div>

        </div>
    )
}

export default ChatBot;