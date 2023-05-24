# filtersignal
Generate a noisy sum of sinsusoids and display it, filter the noisy signal, and then display the filtered signal.

This is a web application written in Go that uses the html/template package to generate the HTML.  The user accesses
the program from a web browser by entering the URL http://127.0.0.1:8080/filtersignal.  The user selects the number of 
samples, the sampling frequency, the signal-to-noise-ratio (SNR), and optionally the filter filename.  From one to five
sinsuoids can be added with the desired Gaussian noise level.  After submission of the form, the plot of the filtered or 
unfiltered signal is displayed.  Successive submits will create a succession of sequences consisting of the desired number
of samples.  A new signal sequence can be generated by clicking on the <i>New Signal Sequence</i> link.  The filtering is 
done using the convolution sum:

![image](https://github.com/thomasteplick/filtersignal/assets/117768679/dc066abd-bff4-4be0-a4c4-057d88a714d2)

where <b>h[n]</b> are the FIR filter coefficients that comprise the impulse response.

<h4>Four sinusoids, signal-to-noise ratio (SNR) 20 dB, 8000 samples and sample rate = 4000 samples/sec.</h4>

![image](https://github.com/thomasteplick/filtersignal/assets/117768679/2fe72dd0-500d-4c7a-88e2-1d4275284173)

<h4>FIR lowpass filter, passband is 0 to 400 Hz, transition band is 400 Hz to 600 Hz, stopband is 600 Hz to 2000 Hz</h4>

Order 57 FIR filter using the Parks-McClellan algorithm and the Remez exchange for equripple error in passband and stopband.
The stopband is greater than 60 dB below the passband.
![image](https://github.com/thomasteplick/filtersignal/assets/117768679/321c8041-0511-455e-bc86-6ba0601b9651)

<h4>Impulse Response FIR lowpass filter, 58 coefficients using the Parks-McClellan algorithm</h4>
  
![image](https://github.com/thomasteplick/filtersignal/assets/117768679/5a839f6f-e0f1-45a7-bcfc-261710f484cb)
  
<h4>Frequency Response FIR lowpass filter, 58 coefficients using the Parks-McClellan algorithm</h4>

![image](https://github.com/thomasteplick/filtersignal/assets/117768679/71ad7b4b-8fd0-4302-b1e5-ceecd9f847dd)

