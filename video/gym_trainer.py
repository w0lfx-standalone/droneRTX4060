from ultralytics import YOLO

def main():
    model = YOLO('yolo26m.pt') 

    
    model.train(
        data='data.yaml',
        epochs=20,
        imgsz=640,
        batch=6,  
        device=0,
        project='runs/detect',
        name='droneRTX',
        exist_ok=True,
        amp=False,      
        workers=0
    )

if __name__ == '__main__':
    main()