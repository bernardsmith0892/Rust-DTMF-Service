use std::{io::{self, Write}};

use cpal::{traits::{HostTrait, StreamTrait}};
use dtmf::DtmfProcessor;
use rodio::DeviceTrait;

pub mod lib;



fn process_input_stream(data: &[f32], info: &cpal::InputCallbackInfo, processor: &mut DtmfProcessor) {
    match processor.process_samples(data, info) {
        Some(char) => {
            print!("{}", char);
            io::stdout().flush().unwrap();
        },
        None => {},
    }
}

fn main() {
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
        move |data, info| process_input_stream(data, info, &mut processor),
        |_| {}
    ).expect("cannot build stream!");

    stream.play().unwrap();

    loop { }
}