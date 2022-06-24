use cpal::{traits::{HostTrait, StreamTrait}, StreamConfig, SampleRate};
use rodio::DeviceTrait;

mod lib;

fn main() {
    let host = cpal::default_host();
    let device = host.default_input_device().expect("no input device available!");
    let config = device.default_input_config().unwrap().config();

    println!("{}, {:?}", device.name().unwrap(), config);
    for supported in device.supported_input_configs().unwrap() {
        println!("{:?}", supported);
    }

    let stream = device.build_input_stream(
        &config, 
        | data: &[f32], info: &cpal::InputCallbackInfo | 
        {
            let mut frequencies: Vec<f32> = Vec::new();
            for freq in lib::DTMF_FREQUENCIES {
                let magnitude = lib::goertzel(freq as f32, 48_000.0, data);
                frequencies.push(magnitude);
            }
            let mean = frequencies.iter().sum::<f32>() / frequencies.len() as f32;
            let std_dev = lib::standard_deviation(&frequencies).unwrap();
            //println!("{:.0?} - {:.4} - {:.4}", frequencies, mean, std_dev);
            //println!("{:?}", lib::get_dtmf_components(&data, 48_000.0));
            //println!("{:?}", lib::decode_dtmf(&data, 48_000.0));
        }, 
        |err| {}
    ).expect("cannot build stream!");

    stream.play().unwrap();

    loop { }
}