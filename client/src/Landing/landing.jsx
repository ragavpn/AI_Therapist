import Navbar from "../Components/Navbar/Navbar"
import cat from '../Images/catpic.png'
import seeMore from '../Images/seemore.png'
import { Link } from "react-router-dom"

function Landing(props) {
    return(
        <div id="landing-page" className="bg-#65647C h-screen">
            <Navbar curPage={props.page}/>
            <div id="content">
                <div id="left-content">
                    <h1>Feeling down, stressed, anxious, or 
                        depressed out of the blue is totally normal.
                    </h1>
                    <h2>
                        Talk to our extensively trained AI chatbot
                        and maybe try understanding your feelings.
                    </h2>
                    <Link id="see-more" to="">
                        <img id="see-more-button" src={seeMore} alt="" />
                    </Link>
                </div>
                <div id="right-content">
                    <img src={cat} alt="" className="cat-image"/>
                </div>

            </div>
        </div>
    )
}

export default Landing;