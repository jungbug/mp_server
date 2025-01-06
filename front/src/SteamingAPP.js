import React, { useState } from "react";

function App() {
  const [currentStream, setCurrentStream] = useState("/video_feed"); // 초기 스트림 URL

  const handleSwitchStream = (to) => {
    // 스트림 전환 요청
    fetch(`http://127.0.0.1:4000/switch?to=${to}`)
      .then((response) => response.text())
      .then((message) => {
        console.log(message);
        //alert(`Switched to ${to === "B" ? "Video B" : "Video A"}`);
      })
      .catch((error) => console.error("Error switching stream:", error));
  };

  const handleResetStream = () => {
    // 프레임 인덱스 초기화 요청
    fetch("http://127.0.0.1:4000/reset")
      .then((response) => response.text())
      .then((message) => {
        console.log(message);
        alert("Frame index reset to 0.");
      })
      .catch((error) => console.error("Error resetting stream:", error));
  };

  return (
    <div style={{ textAlign: "center" }}>
      <h1>MJPEG Stream Viewer</h1>
      <div style={{ marginBottom: "20px" }}>
        <img
          src={`http://127.0.0.1:4000${currentStream}`}
          alt="Video Stream"
          style={{ width: "80%", maxWidth: "600px", border: "2px solid black" }}
          crossOrigin="anonymous" // CORS 문제 해결
        />
      </div>
      <div>
        <button onClick={() => handleSwitchStream("A")}>Switch to Video A</button>
        <button onClick={() => handleSwitchStream("B")} style={{ marginLeft: "10px" }}>
          Switch to Video B
        </button>
        <button onClick={handleResetStream} style={{ marginLeft: "10px" }}>
          Reset Frame Index
        </button>
      </div>
    </div>
  );
}

export default App;
