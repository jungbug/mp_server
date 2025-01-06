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


  //const backend_addr="168.188.128.151:4000"
  const backend_addr = "192.168.50.209:4000"

  const getVideo = () => {
    console.log("get video from server");

    fetch("http://" + backend_addr + "/upload_video", { method: "GET" })
      .then((response) => {
        if (!response.ok) {
          throw new Error("Failed to get video");
        } 
        return response.json();
      })
      .then(data => {
        console.log(data)
        const detVideo=b64toBlob(data.video_result.split('base64,')[1], 'video/mp4')
        handleVideoProcessing(detVideo);
        console.log(detVideo)
        console.log('yes!!!!!!!!')
        console.log("getting local...")
        console.log("Video Source SRC:", videoSrc);
        console.log("hmmm");
      })
      .catch((error) => {
        console.error("Video processing failed:", error);
      });
  };

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

  const uploaderRef = useRef(null);

  const [upProgStatus, setUpProgStatus] = useState('');
  const [isUploaded, setUploaded] = useState(true);
  const [isShowVideo, showVideo] = useState(false);
  const [isShowUpProg, showUpProg] = useState(false);
  const [imageSrc, setImageSrc] = useState([]);

  const [videoSrc, setVideoSrc] = useState('');
  const [videoSrcList, setVideoSrcList] = useState([]);
  const { Dragger } = Upload;

  

  const props4img = {
    name: 'media',
    multiple: false,
    action: backend_addr,
    maxCount:1,
    customRequest: async (req) => {
      const file=req.file
      getBase64(file, (enc) => {
        console.log("uploadingg")
        fetch("http://" + backend_addr + "/upload", {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image: enc })
        })
          .then(res => res.json())
          .then(data => {
            const imageResult = data.image_result;
            if (imageResult) {
              const imgSrc = `data:image/png;base64,${imageResult}`;
              setImageSrc((prevList) => [imgSrc, ...prevList]);

            } else {
              message.error("Image processing failed.");
            }
          });
      })
    
      setVideoSrc(localStorage.getItem("srcVideo"))
    },
    beforeUpload: (file) => {
      const isMP4 = (file.type === 'image/jpg') || (file.type === 'image/png');
      if (!isMP4) {
        message.error(`${file.name} is not a image file`);
      }
      return isMP4 || Upload.LIST_IGNORE;
    }
  };

  useEffect(() => {
    if (isShowVideo && videoSrc) {
      console.log("Video is ready to play:", videoSrc);
    }
  }, [isShowVideo, videoSrc]);
  
  const handleVideoProcessing = (detVideo) => {
    const detVideoUrl = URL.createObjectURL(detVideo);
    localStorage.setItem("testVideo", detVideoUrl);
    setVideoSrc(detVideoUrl);
    console.log("LocalStorage Video Source:", localStorage.getItem("testVideo"));
    showVideo(true);
    console.log(isShowVideo)
  };

  const uploader = ()=>{
    return <><Dragger {...props4img} className='dropbox' ref={uploaderRef}>
    <p className="ant-upload-drag-icon">
    <FileImageOutlined />
    </p>
  </Dragger></>
  }

  const show_image = () => {
    return (
      <div className="imagebox">
        {imageSrc.length > 0 ? (
          imageSrc.map((src, index) => (
            <Image
              key={index}
              src={src}
              alt={`Processed Image ${index + 1}`}
              style={{ marginBottom: '10px', width: '100%', height: 'auto' }}
            />
          ))
        ) : (
          <Typography.Text>No images uploaded yet.</Typography.Text>
        )}
      </div>
    );
  };

  const start_btn = () => {
    return (
      <div className="start_btn">
        <Button
          className="start_btn"
          color="primary"
          variant="solid"
          onClick={getVideo}
        >
          Run process
        </Button>
      </div>
    );
  };

  const control_panel = ()=>{
    return <div className='control_panel'>
      {uploader()}
      {show_image()}
      {start_btn()}
    </div>
  }

  const antIcon = (
    <Tooltip title={upProgStatus} trigger={null} defaultOpen>
    <LoadingOutlined
      style={{
        fontSize: 100,
      }}
      spin
    />
    </Tooltip>
  );

  const renderCards = () => {
    return Array(9).fill(null).map((_, index) => (
      <Card key={index} className="mainCard" bordered={false}>
        {isShowUpProg ? (
          <Spin indicator={antIcon} size="large">
            <div style={{ height: '60vh' }} />
          </Spin>
        ) : (
          <>
            {!isShowVideo ? (
              <Dragger {...getVideo} className="dropbox" ref={uploaderRef}>
                {/* <p className="ant-upload-drag-icon">
                  <PlusOutlined />
                </p> */}
              </Dragger>
            ) : (
              <video
                className="player video-player"
                src={videoSrc}
                autoPlay
                controls
                onLoadedData={() => console.log(`Video ${index + 1} loaded:`, videoSrc)}
              />
            )}
          </>
        )}
      </Card>
    ));
  };
  
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
