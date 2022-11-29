# Rust-DTMF-Service

Rust program that allows you to run remote commands through DTMF sequences. Implements decoding function using a Goertzel filter or correlation.

## Example usage:

```
Select input device:
 0 - "Headset Microphone (Arctis 7 Chat)"
 1 - "CABLE Output (VB-Audio Virtual Cable)"
 2 - "Microphone (Steam Streaming Microphone)"
> 1
Listening for DTMF commands...
28-Nov-2022 23:51:50.419 - 1234#
Output: "Transmission Received!"
28-Nov-2022 23:51:56.740 - 1236#
28-Nov-2022 23:52:01.480 - ABCD
28-Nov-2022 23:52:06.760 - 1234#
Output: "Transmission Received!"
```

## Decoder Performance Comparison using a Channel with Random Noise

```
   Compiling dtmf v0.1.0 (C:\Users\berna\OneDrive\Projects\Rust\dtmf)
    Finished test [unoptimized + debuginfo] target(s) in 1.24s
     Running unittests src\main.rs (target\debug\deps\dtmf-8ea9b36e783c1608.exe)

running 1 test
Trial   Goertzel        Correlation
0       ✔               ✔
1       ✔               ✔
2       ✔               ✔
3       ✔               ✔
4       ✔               ✔
5       ✔               ✔
6       ✔               ✔
7       ✔               ✔
8       ✔               ✔
9       ✔               ✔
10      ✔               ✔
11      ✔               ✔
12      ✔               ✔
13      ✔               ✔
14      ✔               ✔
15      ✔               ✔
16      ✔               ✔
17      ✔               ✔
18      ✔               ✔
19      ✔               ✔
20      ✔               ✔
21      ✔               ✔
22      ✔               ✔
23      ✔               ✔
24      ✔               ✔
25      ✔               ✔
26      ✔               ✔
27      ✔               ✔
28      ✔               ✔
29      ✔               ✔
30      ✔               ✔
31      ✔               ✔
32      ✔               ✔
33      ✔               ✘
34      ✔               ✔
35      ✔               ✔
36      ✔               ✔
37      ✔               ✔
38      ✔               ✘
39      ✔               ✔
40      ✔               ✔
41      ✔               ✔
42      ✔               ✔
43      ✔               ✔
44      ✔               ✔
45      ✔               ✔
46      ✔               ✔
47      ✔               ✔
48      ✔               ✘
49      ✔               ✔
Goertzel: 50 of 50 (100.00%)
Correlation: 47 of 50 (94.00%)
test dtmf::tests::dtmf_multiple_digits_test_noisy ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 5 filtered out; finished in 11.26s
```
