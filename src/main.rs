use std::{collections::VecDeque, sync::Mutex};

use cpal::{traits::{HostTrait, StreamTrait}, StreamConfig, SampleRate, BufferSize};
use rodio::DeviceTrait;

mod lib;

struct DtmfProcessor {
    sample_rate: u32,
    input_buffer: Mutex<Vec<f32>>,
    minimum_n: u32,
}

impl DtmfProcessor {
    pub fn process_input_stream(&mut self, data: &[f32], info: &cpal::InputCallbackInfo) {
        let mut input_buffer = self.input_buffer.lock().unwrap();
        input_buffer.extend(data);

        if input_buffer.len() >= self.minimum_n as usize {
            let mut frequencies: Vec<f32> = Vec::new();
            for freq in lib::DTMF_FREQUENCIES {
                let magnitude = lib::goertzel(freq as f32, self.sample_rate, data);
                frequencies.push(magnitude);
            }
            let mean = frequencies.iter().sum::<f32>() / frequencies.len() as f32;
            let std_dev = lib::standard_deviation(&frequencies).unwrap();
            println!("{:.0?} - {:.4} - {:.4} ({})", frequencies, mean, std_dev, input_buffer.len());
            //println!("{:?}", lib::get_dtmf_components(&data, self.sample_rate));
            //println!("{:?}", lib::decode_dtmf(&data, self.sample_rate));

            input_buffer.clear();
        }
    }
}

fn main() {
    let host = cpal::default_host();
    let device = host.default_input_device().expect("no input device available!");
    let config = device.default_input_config().unwrap().config();

    let mut processor = DtmfProcessor {
        sample_rate: config.sample_rate.0,
        input_buffer: Mutex::new(Vec::new()),
        minimum_n: config.sample_rate.0 / 70,
    };

    println!("{}, {:?}", device.name().unwrap(), config);
    for supported in device.supported_input_configs().unwrap() {
        println!("{:?}", supported);
    }

    let stream = device.build_input_stream(
        &config, 
        move |data, info| processor.process_input_stream(data, info),
        |err| {}
    ).expect("cannot build stream!");

    stream.play().unwrap();

    loop { }
}