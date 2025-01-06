import logo from './logo.svg';
import './App.css';
import { Spin, Layout, Menu, Typography, theme, Col, Row, Card, Slider, Upload, message, Tag, Table, Tooltip, Progress, Button, Segmented, Radio, Image, ConfigProvider, Space } from 'antd';
import { InboxOutlined, LoadingOutlined, CaretRightOutlined, ExpandOutlined, PauseOutlined, FileImageOutlined, PlusOutlined } from '@ant-design/icons';
import { createFFmpeg,  } from "@ffmpeg/ffmpeg";
import React, { useRef, useState, createRef, useEffect } from 'react';


// 돌아가는거 test.mp4 temp_video.mp4
// 안돌아가는거 output.mp4 output_fixed.mp4

// front에서 run process 누르면 back에서 temp video 보내서 띄워주기!! 
function App() {

  function getBase64(file, cb) {
    let reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = function () {
        cb(reader.result)
    };
    reader.onerror = function (error) {
        console.log('Error: ', error);
    };
  }

  const b64toBlob = (b64Data, contentType='', sliceSize=512) => {
    const byteCharacters = atob(b64Data);
    const byteArrays = [];
  
    for (let offset = 0; offset < byteCharacters.length; offset += sliceSize) {
      const slice = byteCharacters.slice(offset, offset + sliceSize);
  
      const byteNumbers = new Array(slice.length);
      for (let i = 0; i < slice.length; i++) {
        byteNumbers[i] = slice.charCodeAt(i);
      }
  
      const byteArray = new Uint8Array(byteNumbers);
      byteArrays.push(byteArray);
    }
  
    const blob = new Blob(byteArrays, {type: contentType});
    return blob;
  }


  const ffmpeg = createFFmpeg({
    log: false,
  });
  (async () => {
      await ffmpeg.load();
  })();


  const backend_addr = "127.0.0.1:4000"

  const [currentStream, setCurrentStream] = useState("/video_feed"); // 초기 스트림 URL
  const uploaderRef = useRef(null);
  const [imageSrc, setImageSrc] = useState([]);
  const { Dragger } = Upload;

  const props4img = {
    name: 'media',
    multiple: false,
    maxCount: 1,
    customRequest: async (req) => {
      const file = req.file;
  
      // Base64로 변환하여 상태에 저장
      getBase64(file, (enc) => {
        console.log("Image uploaded locally");
        const imgSrc = enc; // Base64 이미지 URL
        setImageSrc((prevList) => [imgSrc, ...prevList]);
      });
    },
    beforeUpload: (file) => {
      const isImage = (file.type === 'image/jpg') || (file.type === 'image/png') || (file.type === 'image/jpeg');
      if (!isImage) {
        message.error(`${file.name} is not a valid image file`);
      }
      return isImage || Upload.LIST_IGNORE;
    },
  };

  const uploader = ()=>{
    return <><Dragger {...props4img} className='dropbox' ref={uploaderRef}>
    <p className="ant-upload-drag-icon">
    <FileImageOutlined />
    </p>
  </Dragger></>
  }

  const handleDeleteImage = (indexToDelete) => {
    setImageSrc((prevList) => prevList.filter((_, index) => index !== indexToDelete));
  };

  const show_image = () => {
    return (
      <div className="imagebox">
        {imageSrc.length > 0 ? (
          imageSrc.map((src, index) => (
            <React.Fragment key={index}>
              <Image
                key={index}
                src={src}
                alt={`Processed Image ${index + 1}`}
                style={{ marginBottom: '10px', width: '100%', height: 'auto' }}
              />
              <Button className="deleteButton"
                type="primary"
                danger
                size="small"
                onClick={() => handleDeleteImage(index)}
                style={{
                  position: 'relative',
                  top: '-35px',
                  right: '10px',
                  zIndex: 10, // 이미지 위에 표시
                  float: 'right', // 오른쪽 정렬
                }}
                >
                Delete
              </Button>
            </React.Fragment>
          ))
        ) : (
          <Typography.Text>No images uploaded yet.</Typography.Text>
        )}
      </div>
    );
  };

  const handleSendImagesToBackend = () => {
    if (imageSrc.length === 0) {
      message.error("No images to send.");
      return;
    }
  
    const imagesPayload = imageSrc.map((image) => {
      // 이미지에서 base64 데이터만 추출
      return image.split(',')[1];
    });
  
    fetch("http://127.0.0.1:4000/upload", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ images: imagesPayload }),
    })
      .then((response) => response.json())
      .then((data) => {
        console.log("Server Response:", data);
        if (data.success) {
          message.success("Images processed successfully.");
          handleSwitchStream("ai"); // 성공 시 스트림 전환
        } else {
          message.error("Failed to process images.");
        }
      })
      .catch((error) => {
        console.error("Error sending images to backend:", error);
        message.error("An error occurred while sending images.");
      });
  };

  // 이미지 전송 및 트래킹 시작
  const handleTracking = async () => {
    const success = handleSendImagesToBackend(); // 이미지 전송 후 결과 대기
    
  };

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


  const switchToTracking = () => {
    return (
      <div className="start_btn">
        <Button
          className="start_btn"
          color="primary"
          variant="solid"
          //onClick={()=>handleSwitchStream("ai")}
          onClick={handleTracking}
        >
          Track
        </Button>
      </div>
    );
  };

  const switchToOriginal = () => {
    return (
      <div className="start_btn">
        <Button
          className="start_btn"
          color="primary"
          variant="solid"
          onClick={()=>handleSwitchStream("original")}
        >
          Original Video
        </Button>
      </div>
    );
  };

  const control_panel = ()=>{
    return <div className='control_panel'>
      {uploader()}
      {show_image()}
      {switchToTracking()}    
      {switchToOriginal()}
    </div>
  }

  const renderCards = () => {
    const serverAddresses = [
      // 여기 바꾸셈 서버주소
      "http://127.0.0.1:4001",
      "http://127.0.0.1:4004",
      "http://127.0.0.1:4006",
      // "http://127.0.0.1:4000",
      // "http://127.0.0.1:4000",
      // "http://127.0.0.1:4000",
      // "http://127.0.0.1:4000",
      // "http://127.0.0.1:4000",
      // "http://127.0.0.1:4000",
    ];
  
    return serverAddresses.map((address, index) => (
      <Card key={index} className="mainCard" bordered={false}>
        <img
          className="video-player"
          src={`${address}${currentStream}`} // 각 서버 주소를 src로 사용
          alt={`Video Stream ${index + 1}`}
          crossOrigin="anonymous" // CORS 문제 해결
        />
      </Card>
    ));
  };
  

  // const renderCards = () => {
  //   return Array(9).fill(null).map((_, index) => (
  //     <Card key={index} className="mainCard" bordered={false}>
  //       <img className="video-player"
  //         src={`http://127.0.0.1:4000${currentStream}`}
  //         alt="Video Stream"
  //         crossOrigin="anonymous" // CORS 문제 해결
  //       />
  //     </Card>
  //   ));
  // };

  const show_video = () => {
    return <div className="videobox">{renderCards()}</div>;
  };
  
  const video_panel = ()=>{
    return <div className='video_panel'>
      {show_video()}
    </div>
  }

  return (
    <div className="App">
        <ConfigProvider
          theme={{
            token: {
              // Seed Token
              colorPrimary: '#ffaf96',

              // Alias Token
              colorBgContainer: '#f6ffed',
            },
          }}
        >
      {control_panel()}
      {video_panel()}
      </ConfigProvider>
    </div>
  );
}

export default App;
