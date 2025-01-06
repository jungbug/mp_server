import logo from './logo.svg';
import './App.css';
import { Spin, Layout, Menu, Typography, theme, Col, Row, Card, Slider, Upload, message, Tag, Table, Tooltip, Progress, Button, Segmented, Radio, Image, ConfigProvider, Space } from 'antd';
import { InboxOutlined, LoadingOutlined, CaretRightOutlined, ExpandOutlined, PauseOutlined, FileImageOutlined, PlusOutlined } from '@ant-design/icons';
import { createFFmpeg,  } from "@ffmpeg/ffmpeg";
import React, { useRef, useState, createRef, useEffect } from 'react';



function App() {


  //const backend_addr="168.188.128.151:4000"
  const backend_addr = "192.168.50.209:4000"

  const [statusMessage, setStatusMessage] = useState(null); // 서버 응답 메시지 저장

  const fetchStatus = () => {
    console.log("Fetching status from server...");
    setStatusMessage("Loading...");

    fetch("http://" + backend_addr + "/get_status", { method: "GET" })
      .then((response) => {
        if (!response.ok) {
          throw new Error("Failed to fetch status");
        }
        return response.json();
      })
      .then((data) => {
        console.log("Server Response:", data);
        setStatusMessage(data.status || "No message received"); // 서버 응답 상태 저장
        message.success(`Video Processed : ${data.status || "No message received"}`);
      })
      .catch((error) => {
        console.error("Error fetching status:", error);
        setStatusMessage("Error fetching status"); // 에러 메시지 설정
        message.success(`Server says: ${error || "Error fetching status"}`);
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
      // showUpProg(true)
      const file=req.file
      
      getBase64(file, (enc) => {
        console.log("uploadingg")
        // console.log(enc)
        fetch("http://" + backend_addr + "/upload", {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image: enc })
        })
          .then(res => res.json())
          .then(data => {

            // console.log(data.image_result)

            const imageResult = data.image_result; // Base64 이미지 데이터
            if (imageResult) {
              const imgSrc = `data:image/png;base64,${imageResult}`; // src로 변환
              // setImageSrc(imgSrc); // 상태에 저장하여 렌더링
              setImageSrc((prevList) => [imgSrc, ...prevList]); // 기존 배열에 새로운 이미지 추가

            } else {
              message.error("Image processing failed.");
            }

            // const detVideo=b64toBlob(data['output_video'], 'image/jpg')
            // console.log(detVideo)
            // localStorage.setItem("detVideo", URL.createObjectURL(detVideo));

            // setUpProgPercent(100);
            // showVideo(true);
            // showUpProg(false);
            // setUploaded(true);
          });
      })

      
      setVideoSrc(localStorage.getItem("srcVideo"))
      // videoRef.current.srcObject=localStorage.getItem("detVideo")
      
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


  const props = {
    name: 'media',
    multiple: false,
    action: backend_addr,
    maxCount:1,
    customRequest: async (req) => {
      // showUpProg(true)
      const file=req.file
      
      localStorage.setItem("srcVideowantoshow",URL.createObjectURL(file))

      // setUpProgPercent(20)
      // setUpProgStatus("Loading the file...")
      const srcFile=await file.arrayBuffer()
      
      ffmpeg.FS('writeFile', 'video.mp4', new Uint8Array(srcFile))

      // setUpProgPercent(30)
      // setUpProgStatus("Processing the file...")
      
      await ffmpeg.run('-i', 'video.mp4', '-ss', '0', '-t', '10', '1.mp4')
      
      const data = ffmpeg.FS('readFile','1.mp4')

      // setUpProgPercent(70)
      // setUpProgStatus("Working on AI...")

      // console.log(PRatioRef.current.value)
      console.log(file)
      getBase64(file, (enc) => {
        console.log(enc)
        fetch("http://" + backend_addr + "/upload", {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ video: enc })
        })
          .then(res => res.json())
          .then(data => {

            console.log(data)

            
            const detVideo=b64toBlob(data.video_result.split('base64,')[1], 'video/mp4')
            handleVideoProcessing(detVideo);
            console.log(detVideo)
            console.log('yes!!!!!!!!')
            //localStorage.setItem("testVideo", URL.createObjectURL(detVideo));
            
            //setVideoSrcList((prevList) => [videoSrcList, ...prevList]);
            //showVideo(true);
            //console.log(isShowVideo);
            
            // setUpProgPercent(100);
            // showVideo(true);
            // showUpProg(false);
            // setUploaded(true);
            console.log("getting local...")
            setVideoSrc(localStorage.getItem("srcVideowantoshow"))
            console.log("Video Source SRC:", videoSrc);
            // videoRef.current.srcObject=localStorage.getItem("detVideo")
            // console.log("Video Source:", videoSrc);
            console.log("hmmm");
          })
          .catch((error) => {
            console.error("Video processing failed:", error);
          });
          
      })
      
      console.log("제발 : ", isShowVideo)
      // showVideo(true);
      // showUpProg(true)
      // setUploaded(true);
      
      
    },
    beforeUpload: (file) => {
      const isMP4 = file.type === 'video/mp4';
      if (!isMP4) {
        message.error(`${file.name} is not a mp4 file`);
      }
      return isMP4 || Upload.LIST_IGNORE;
    }
  };


  const uploader = ()=>{
    return <><Dragger {...props4img} className='dropbox' ref={uploaderRef}>
    <p className="ant-upload-drag-icon">
    <FileImageOutlined />
    </p>
  </Dragger></>
  }

  // const show_image = ()=>{
  //   return <div className='imagebox'><Image
  //   src="test.jpg"
  //   />
  //   <Image
  //   src="person.png"
  //   />
  //   <Image
  //   src="person.png"
  //   />
  //   <Image
  //   src="person.png"
  //   />
  //   <Image
  //   src="person.png"
  //   />
  //   <Image
  //   src="person.png"
  //   />
  //   <Image
  //   src="person.png"
  //   />
  //   </div>
  // }
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

  

  // const start_btn = ()=>{
  //   return <div className='start_btn'>
  //     <Button className='start_btn' color="primary" variant="solid">Run process</Button>
  //     </div>
  // }

  const start_btn = () => {
    return (
      <div className="start_btn">
        <Button
          className="start_btn"
          color="primary"
          variant="solid"
          onClick={fetchStatus}
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

  const show_video = ()=>{
    return <div className='videobox'>
      <Card className='mainCard'  bordered={false}>
          {(()=>{
            const upProg=isShowUpProg?<>
                  <Spin indicator={antIcon} size='large'><div style={{height:'60vh'}}/></Spin>
                </>
              :
              <Dragger {...props} className='dropbox' ref={uploaderRef}>
                <p className="ant-upload-drag-icon">
                <PlusOutlined />
                </p>
              </Dragger>
              
            
            return isShowVideo? 
            <video 
                  className='player' 
                  src={videoSrc}
                  onLoadedData={() => console.log("Video loaded:", videoSrc)} />
                : 
                upProg})()}
      </Card>
      {/* <Card className='mainCard'  bordered={false}>
          {(()=>{
            const upProg=isShowUpProg?<>
                  <Spin indicator={antIcon} size='large'><div style={{height:'60vh'}}/></Spin>
                </>
              :
              <Dragger {...props} className='dropbox' ref={uploaderRef}>
                <p className="ant-upload-drag-icon">
                <PlusOutlined />
                </p>
              </Dragger>
              
            
            return isShowVideo? 
            <video 
                  className='player' 
                  src={videoSrc} />
                : 
                upProg})()}
      </Card>
      <Card className='mainCard'  bordered={false}>
          {(()=>{
            const upProg=isShowUpProg?<>
                  <Spin indicator={antIcon} size='large'><div style={{height:'60vh'}}/></Spin>
                </>
              :
              <Dragger {...props} className='dropbox' ref={uploaderRef}>
                <p className="ant-upload-drag-icon">
                <PlusOutlined />
                </p>
              </Dragger>
              
            
            return isShowVideo? 
            <video 
                  className='player' 
                  src={videoSrc} />
                :
                upProg})()}
      </Card>
      <Card className='mainCard'  bordered={false}>
          {(()=>{
            const upProg=isShowUpProg?<>
                  <Spin indicator={antIcon} size='large'><div style={{height:'60vh'}}/></Spin>
                </>
              :
              <Dragger {...props} className='dropbox' ref={uploaderRef}>
                <p className="ant-upload-drag-icon">
                <PlusOutlined />
                </p>
              </Dragger>
              
            
            return isShowVideo? 
            <video 
                  className='player' 
                  src={videoSrc} />
                :
                upProg})()}
      </Card>
      <Card className='mainCard'  bordered={false}>
          {(()=>{
            const upProg=isShowUpProg?<>
                  <Spin indicator={antIcon} size='large'><div style={{height:'60vh'}}/></Spin>
                </>
              :
              <Dragger {...props} className='dropbox' ref={uploaderRef}>
                <p className="ant-upload-drag-icon">
                <PlusOutlined />
                </p>
              </Dragger>
              
            
            return isShowVideo? 
            <video 
                  className='player' 
                  src={videoSrc} />
                :
                upProg})()}
      </Card>
      <Card className='mainCard'  bordered={false}>
          {(()=>{
            const upProg=isShowUpProg?<>
                  <Spin indicator={antIcon} size='large'><div style={{height:'60vh'}}/></Spin>
                </>
              :
              <Dragger {...props} className='dropbox' ref={uploaderRef}>
                <p className="ant-upload-drag-icon">
                <PlusOutlined />
                </p>
              </Dragger>
              
            
            return isShowVideo? 
            <video 
                  className='player' 
                  src={videoSrc} />
                :
                upProg})()}
      </Card>
      <Card className='mainCard'  bordered={false}>
          {(()=>{
            const upProg=isShowUpProg?<>
                  <Spin indicator={antIcon} size='large'><div style={{height:'60vh'}}/></Spin>
                </>
              :
              <Dragger {...props} className='dropbox' ref={uploaderRef}>
                <p className="ant-upload-drag-icon">
                <PlusOutlined />
                </p>
              </Dragger>
              
            
            return isShowVideo? 
            <video 
                  className='player' 
                  src={videoSrc} />
                :
                upProg})()}
      </Card>
      <Card className='mainCard'  bordered={false}>
          {(()=>{
            const upProg=isShowUpProg?<>
                  <Spin indicator={antIcon} size='large'><div style={{height:'60vh'}}/></Spin>
                </>
              :
              <Dragger {...props} className='dropbox' ref={uploaderRef}>
                <p className="ant-upload-drag-icon">
                <PlusOutlined />
                </p>
              </Dragger>
              
            
            return isShowVideo? 
            <video 
                  className='player' 
                  src={videoSrc} />
                :
                upProg})()}
      </Card>
      <Card className='mainCard'  bordered={false}>
          {(()=>{
            const upProg=isShowUpProg?<>
                  <Spin indicator={antIcon} size='large'><div style={{height:'60vh'}}/></Spin>
                </>
              :
              <Dragger {...props} className='dropbox' ref={uploaderRef}>
                <p className="ant-upload-drag-icon">
                <PlusOutlined />
                </p>
              </Dragger>
              
            
            return isShowVideo? 
            <video 
                  className='player' 
                  src={videoSrc} />
                :
                upProg})()}
      </Card> */}
    </div>
  }
  
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
