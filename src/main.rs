use std::{io::{self, Write}};

use cpal::{traits::{HostTrait, StreamTrait}};
use dtmf::DtmfProcessor;
use rodio::DeviceTrait;

pub mod lib;



fn process_input_stream(data: &[f32], processor: &mut DtmfProcessor) {
    match processor.process_samples(data) {
        Some(char) => {
            print!("{}", char);
            io::stdout().flush().unwrap();
        },
        None => {},
    }
}

fn main() {
    //let sine_hz = |n: u32, freq: f32, samp_rate: f32| 
      //f32::sin(freq * 2.0 * std::f32::consts::PI * (n as f32) / samp_rate);

    //let sample_rate: u32 = 1024; 
    //let test_signal: Vec<f32> = (0..102)
        //.map(|n| sine_hz(n, 1000.0, sample_rate as f32))
        //.collect();
    
    //// Test magnitude of two frequencies
    //for freq in (500..1500).step_by(10) {
        //let power = dtmf::goertzel(freq as f32, sample_rate, &test_signal);
        //println!("{} ,{}", freq, power);
    //}


    let host = cpal::default_host();
    let devices = host.input_devices().expect("no input devices available!");
    println!("Select input device:");
    for (i, device) in devices.enumerate() {
        println!("\t{} - {:?}", i, device.name().unwrap());
    }

    let mut input_string = String::new();
    std::io::stdin().read_line(&mut input_string).unwrap();
    let device_index = input_string.trim().parse::<i32>().expect("must provide a number!");


    let device = host.input_devices().unwrap().nth(device_index as usize).expect("cannot use this device!");
    let config = device.default_input_config().unwrap().config();

    let mut processor = dtmf::DtmfProcessor::new(config.sample_rate.0, config.channels); 

    let stream = device.build_input_stream(
        &config, 
        move |data, _| process_input_stream(data, &mut processor),
        |_| {}
    ).expect("cannot build stream!");

    stream.play().unwrap();


    loop { }
}