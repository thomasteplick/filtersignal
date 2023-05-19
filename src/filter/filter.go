/*
	Perform the convolution sum for a signal s[n] with the given filter impulse response h[n]:
	s[n]*h[n], where * is the convolution sum.  The user can also view the unfiltered signal.
	User selects signals, SNR, number of samples, sample rate, sine frequencies and amplitudes,
	and filter file from the HTML form  in the web browser.

	Multiple SUBMITs of the HTML form will display a continuous filtered sequence of the signals.
	The state of the sequence is saved at the end of each submit.

	This program simulates using Direct Memory Access (DMA) with interrupts on a Digital Signal Processor (DSP).
	Multiple goroutines allow generating the signal and filtering the signal concurrently.  Mulitple-
	core processors will optimize this approach.  The signal generator goroutine will produce samples
	of the signal and synchronize with the filter goroutine when done creating a block of the samples.
	These will be placed in one buffer.  The filter routine will process the samples in that buffer.  In
	the meantime, the generator goroutine will start filling another buffer with samples.  It will then
	synchronize with the filter goroutine when the filter is ready to accept the new block.  The cycle repeats
	until all the samples are produced and filtered.  This is a ping-pong technique where the buffers alternate
	between generating and filtering.  Synchronization is done using unbuffered channels which behave as semaphores.
*/

package main

import (
	"bufio"
	"fmt"
	"html/template"
	"log"
	"math"
	"math/rand"
	"net/http"
	"os"
	"path"
	"strconv"
	"sync"
)

const (
	rows                        = 300 // #rows in grid
	columns                     = 300 // #columns in grid
	block                       = 512 // size of buf1 and buf2, chunks of data to process
	patternFilterSignal         = "/filtersignal"
	tmplfiltersignal            = "templates/filtersignal.html"
	addr                        = "127.0.0.1:8080"  // http server listen address
	xlabels                     = 11                // # labels on x axis
	ylabels                     = 11                // # labels on y axis
	dataDir                     = "data/"           // directory for the signal state
	stateFile                   = "filterstate.txt" // last M-1 filtered outputs from prev, ampl & freq of signal, time of last sample
	twoPi               float64 = 2.0 * math.Pi
	signal                      = "signal.txt" //  signal
)

type Attribute struct {
	Freq     string
	Ampl     string
	FreqName string
	AmplName string
}

// Type to contain all the HTML template actions
type PlotT struct {
	Grid       []string    // plotting grid
	Status     string      // status of the plot
	Xlabel     []string    // x-axis labels
	Ylabel     []string    // y-axis labels
	Filename   string      // filename of filter coefficients
	SampleFreq string      // data sampling rate in Hz
	Samples    string      // complex samples in data file
	SNR        string      // signal-to-noise ratio
	Sines      []Attribute // frequency and amplitude of the sines
}

// properties of the sinusoids for generating the signal
type Sine struct {
	freq int
	ampl float64
}

// previous sample block properties used for generating/filtering current block
type FilterState struct {
	lastFiltered   []float64 // last M-1 incomplete filtered samples from previous block
	lastSampleTime float64   // from previous block of samples, in seconds
}

type FilterSignal struct {
	sema1            chan int // semaphores to synchronize access to the ping-pong buffers
	sema2            chan int
	wg               sync.WaitGroup
	buf              [][]float64   // ping-pong buffer
	done             chan struct{} // generator Signal to the filter when all samples generated
	samples          int           // total number of samples per submit
	samplesGenerated int           // number of samples generated so far for this submit
	sampleFreq       int           // sample frequency in Hz
	snr              int           // signal to noise ratio in dB
	sines            []Sine        // sinusoids to generate in the signalsignal
	FilterState                    // used by current sample block from previous sample block
	filterCoeff      []float64     // filter coefficients
	filterfile       string        // name of the FIR filter file
	Endpoints                      // embedded struct
}

// Type to hold the minimum and maximum data values
type Endpoints struct {
	xmin float64
	xmax float64
	ymin float64
	ymax float64
}

var (
	filterSignalTmpl *template.Template
)

// init parses the HTML template file
func init() {
	filterSignalTmpl = template.Must(template.ParseFiles(tmplfiltersignal))
}

// fillBuf populates the buffer with signal samples
func (fs *FilterSignal) fillBuf(n int, noiseSD float64) int {
	// get last sample time from previous block
	// fill buf n with noisy sinusoids with SNR given
	// Sum the sinsuoids and noise and insert into the buffer

	// Determine how many samples we need to generate
	howMany := block
	toGo := fs.samples - fs.samplesGenerated
	if toGo < block {
		howMany = toGo
	}

	delta := 1.0 / float64(fs.sampleFreq)
	var t float64
	t = fs.lastSampleTime
	for i := 0; i < howMany; i++ {
		sinesum := 0.0
		for _, sig := range fs.sines {
			omega := twoPi * float64(sig.freq)
			sinesum += float64(sig.ampl) * math.Sin(omega*t)
		}
		sinesum += noiseSD * rand.NormFloat64()
		fs.buf[n][i] = sinesum
		t += delta
	}
	// Save the next sample time for next block of samples
	fs.lastSampleTime = t
	fs.samplesGenerated += howMany
	return howMany
}

// findEndpoints finds the minimum and maximum filter values
func (ep *Endpoints) findEndpoints(input *bufio.Scanner, xmax float64) {
	ep.xmax = xmax
	ep.xmin = 0.0
	ep.ymax = -math.MaxFloat64
	ep.ymin = math.MaxFloat64
	for input.Scan() {
		value := input.Text()
		var (
			y   float64
			err error
		)

		if y, err = strconv.ParseFloat(value, 64); err != nil {
			fmt.Printf("findEndpoints string %s conversion to float error: %v\n", value, err)
			continue
		}

		if y > ep.ymax {
			ep.ymax = y
		}
		if y < ep.ymin {
			ep.ymin = y
		}
	}
}

// gridFillInterp inserts the data points in the grid and draws a straight line between points
func (fs *FilterSignal) gridFillInterp(plot *PlotT) error {
	var (
		x            float64 = 0.0
		y            float64 = 0.0
		prevX, prevY float64
		err          error
		xscale       float64
		yscale       float64
		endpoints    Endpoints
		input        *bufio.Scanner
		timeStep     float64 = 1.0 / float64(fs.sampleFreq)
	)
	// Open file
	f, err := os.Open(path.Join(dataDir, signal))
	if err != nil {
		fmt.Printf("Error opening %s: %v\n", signal, err.Error())
	}
	// Mark the data x-y coordinate online at the corresponding
	// grid row/column.
	input = bufio.NewScanner(f)

	endpoints.findEndpoints(input, fs.lastSampleTime)
	fs.Endpoints = endpoints

	f.Close()
	f, err = os.Open(path.Join(dataDir, signal))
	if err != nil {
		fmt.Printf("Error opening %s: %v\n", signal, err.Error())
		return err
	}
	defer f.Close()
	input = bufio.NewScanner(f)

	// Get first sample
	input.Scan()
	value := input.Text()

	if y, err = strconv.ParseFloat(value, 64); err != nil {
		fmt.Printf("gridFillInterp first sample string %s conversion to float error: %v\n", value, err)
		return err
	}

	plot.Grid = make([]string, rows*columns)

	// This cell location (row,col) is on the line
	row := int((endpoints.ymax-y)*yscale + .5)
	col := int((x-endpoints.xmin)*xscale + .5)
	plot.Grid[row*columns+col] = "online"

	prevX = x
	prevY = y

	// Scale factor to determine the number of interpolation points
	lenEPy := endpoints.ymax - endpoints.ymin
	lenEPx := endpoints.xmax - endpoints.xmin

	// Continue with the rest of the points in the file
	for input.Scan() {
		x += timeStep
		value = input.Text()
		if y, err = strconv.ParseFloat(value, 64); err != nil {
			fmt.Printf("gridFillInterp the rest of file string %s conversion to float error: %v\n", value, err)
			return err
		}

		// This cell location (row,col) is on the line
		row := int((endpoints.ymax-y)*yscale + .5)
		col := int((x-endpoints.xmin)*xscale + .5)
		plot.Grid[row*columns+col] = "online"

		// Interpolate the points between previous point and current point

		/* lenEdge := math.Sqrt((x-prevX)*(x-prevX) + (y-prevY)*(y-prevY)) */
		lenEdgeX := math.Abs((x - prevX))
		lenEdgeY := math.Abs(y - prevY)
		ncellsX := int(columns * lenEdgeX / lenEPx) // number of points to interpolate in x-dim
		ncellsY := int(rows * lenEdgeY / lenEPy)    // number of points to interpolate in y-dim
		// Choose the biggest
		ncells := ncellsX
		if ncellsY > ncells {
			ncells = ncellsY
		}

		stepX := (x - prevX) / float64(ncells)
		stepY := (y - prevY) / float64(ncells)

		// loop to draw the points
		interpX := prevX
		interpY := prevY
		for i := 0; i < ncells; i++ {
			row := int((endpoints.ymax-interpY)*yscale + .5)
			col := int((interpX-endpoints.xmin)*xscale + .5)
			plot.Grid[row*columns+col] = "online"
			interpX += stepX
			interpY += stepY
		}

		// Update the previous point with the current point
		prevX = x
		prevY = y
	}
	return nil
}

// filterBuf filters the samples with the given filter coefficients
// index is the buffer to use, 1 or 2, nsamples is the number of samples to filter
func (fs *FilterSignal) filterBuf(index int, nsamples int, f *os.File) {
	// have the last M-1 filter outputs from previous block in FilterSignal
	// loop over the samples from the generator and apply the filter coefficients
	// in a convolution sum

	// Incorporate the previous block partially filtered outputs
	m := len(fs.filterCoeff)
	end := m - 1
	if nsamples < m-1 {
		end = nsamples
	}
	for n := 0; n < end; n++ {
		sum := fs.lastFiltered[n]
		for k := 0; k <= n; k++ {
			sum += fs.buf[index][k] * fs.filterCoeff[n-k]
		}
		fmt.Fprintf(f, "%f\n", sum)
	}

	// This is the last block and it has no more samples since nsamples <= m-1
	if end == nsamples {
		return
	}

	for n := end; n < nsamples; n++ {
		sum := 0.0
		for k := n - m + 1; k <= n; k++ {
			sum += fs.buf[index][k] * fs.filterCoeff[n-k]
		}
		fmt.Fprintf(f, "%f\n", sum)
	}

	// Generate the partially filtered outputs used in next block
	i := 0
	if nsamples == block {
		for n := block - m + 1; n < block; n++ {
			sum := 0.0
			for k := n; k < block; k++ {
				sum += fs.buf[index][k] * fs.filterCoeff[m-k+n-1]
			}
			fs.lastFiltered[i] = sum
			i++
		}
	}
}

// nofilterBuf saves the signal to a file.  It is not modified.
// index is the buffer to use, 1 or 2, nsamples is the number of samples to filter
func (fs *FilterSignal) nofilterBuf(index int, nsamples int, f *os.File) {
	for n := 0; n < nsamples; n++ {
		fmt.Fprintf(f, "%f\n", fs.buf[index][n])
	}
}

// generate creates the noisy signal, it is the producer or generator
func (fs *FilterSignal) generate(r *http.Request) error {

	// get SNR, sine frequencies and sine amplitudes
	temp := r.FormValue("snr")
	if len(temp) == 0 {
		return fmt.Errorf("missing SNR for Sum of Sinsuoids")
	}
	snr, err := strconv.Atoi(temp)
	if err != nil {
		return err
	}

	fs.snr = snr
	var (
		maxampl  float64  = 0.0 // maximum sine amplitude
		freqName []string = []string{"SSfreq1", "SSfreq2",
			"SSfreq3", "SSfreq4", "SSfreq5"}
		ampName []string = []string{"SSamp1", "SSamp2", "SSamp3",
			"SSamp4", "SSamp5"}
	)

	// get the sine frequencies and amplitudes, 1 to 5 possible
	for i, name := range freqName {
		a := r.FormValue(ampName[i])
		f := r.FormValue(name)
		if len(a) > 0 && len(f) > 0 {
			freq, err := strconv.Atoi(f)
			if err != nil {
				return err
			}
			ampl, err := strconv.ParseFloat(a, 64)
			if err != nil {
				return err
			}
			fs.sines = append(fs.sines, Sine{freq: freq, ampl: ampl})
			if ampl > maxampl {
				maxampl = ampl
			}
		}
	}

	// Require at least one sine to create
	if len(fs.sines) == 0 {
		return fmt.Errorf("enter frequency and amplitude of 1 to 5 sinsuoids")
	}

	// Calculate the noise standard deviation using the SNR and maxampl
	ratio := math.Pow(10.0, float64(snr)/10.0)
	noiseSD := math.Sqrt(0.5 * float64(maxampl) * float64(maxampl) / ratio)

	// increment wg
	fs.wg.Add(1)

	// launch a goroutine to generate samples
	go func() {
		// if all samples generated, signal filter on done semaphore and return
		defer fs.wg.Done()
		defer func() {
			fs.done <- struct{}{}
		}()

		// loop to generate a block of signal samples
		// signal the filter when done with each block of samples
		// block on a semaphore until filter goroutine is available
		for {
			n := fs.fillBuf(0, noiseSD)
			fs.sema1 <- n
			if n < block {
				return
			}
			n = fs.fillBuf(1, noiseSD)
			fs.sema2 <- n
			if n < block {
				return
			}
		}
	}()

	return nil
}

// filter processes the noisy signal, it is the consumer
func (fs *FilterSignal) filter(r *http.Request, plot *PlotT) error {
	// increment wg
	fs.wg.Add(1)

	// if no filter file specified, pass the samples unchanged
	filename := r.FormValue("filter")
	if len(filename) == 0 {
		f2, _ := os.Create(path.Join(dataDir, signal))

		// launch a goroutine to no-filter generator signal
		// select on the generator semaphores and the done channel
		go func() {
			defer f2.Close()
			defer fs.wg.Done()
			for {
				select {
				case n := <-fs.sema1:
					fs.nofilterBuf(0, n, f2)
				case n := <-fs.sema2:
					fs.nofilterBuf(1, n, f2)
				case <-fs.done:
					// if done signal from generator, save state, then return
					// save the Signal state:  time of last sample
					f, err := os.Create(path.Join(dataDir, stateFile))
					if err != nil {
						fmt.Printf("Create %s save state error: %v\n", stateFile, err)
					} else {
						defer f.Close()
						fmt.Fprintf(f, "%f\n", fs.lastSampleTime)
					}
					return
				}
			}
		}()
	} else {
		// get filter coefficients from file specified by user
		f, err := os.Open(path.Join(dataDir, filename))
		if err != nil {
			return fmt.Errorf("open %s error %v", filename, err)
		}
		defer f.Close()
		input := bufio.NewScanner(f)
		for input.Scan() {
			line := input.Text()
			h, err := strconv.ParseFloat(line, 64)
			if err != nil {
				return fmt.Errorf("filter coefficient conversion error: %v", err)
			}
			fs.filterCoeff = append(fs.filterCoeff, h)
		}

		// allocate memory for filtered output from previous submit
		fs.lastFiltered = make([]float64, len(fs.filterCoeff))

		f2, _ := os.Create(path.Join(dataDir, signal))

		// launch a goroutine to filter generator signal
		// select on the generator semaphores and the done channel
		go func() {
			defer f2.Close()
			defer fs.wg.Done()
			for {
				select {
				case n := <-fs.sema1:
					fs.filterBuf(0, n, f2)
				case n := <-fs.sema2:
					fs.filterBuf(1, n, f2)
				case <-fs.done:
					// if done signal from generator, save state, then return
					// save the Signal state:  time of last sample and incomplete filtered output
					f, err := os.Create(path.Join(dataDir, stateFile))
					if err != nil {
						fmt.Printf("Create %s save state error: %v\n", stateFile, err)
					} else {
						defer f.Close()
						fmt.Fprintf(f, "%f\n", fs.lastSampleTime)
						for _, lf := range fs.lastFiltered {
							fmt.Fprintf(f, "%f\n", lf)
						}
					}
					return
				}
			}
		}()
	}

	return nil
}

// showSineTable displays an empty sine table
func showSineTable(plot *PlotT) {
	// Fill in the table of frequencies and their amplitudes
	var (
		freqName []string = []string{"SSfreq1", "SSfreq2",
			"SSfreq3", "SSfreq4", "SSfreq5"}
		ampName []string = []string{"SSamp1", "SSamp2", "SSamp3",
			"SSamp4", "SSamp5"}
	)

	// show the sine table even if empty, otherwise it will never be filled in
	plot.Sines = make([]Attribute, len(freqName))
	for i := range freqName {
		plot.Sines[i] = Attribute{
			FreqName: freqName[i],
			AmplName: ampName[i],
			Freq:     "",
			Ampl:     "",
		}
	}
}

// label the plot and execute the PlotT on the HTML template
func (fs *FilterSignal) labelExec(w http.ResponseWriter, plot *PlotT, endpoints *Endpoints) {

	plot.Xlabel = make([]string, xlabels)
	plot.Ylabel = make([]string, ylabels)

	// Construct x-axis labels
	incr := (endpoints.xmax - endpoints.xmin) / (xlabels - 1)
	x := endpoints.xmin
	// First label is empty for alignment purposes
	for i := range plot.Xlabel {
		plot.Xlabel[i] = fmt.Sprintf("%.2f", x)
		x += incr
	}

	// Construct the y-axis labels
	incr = (endpoints.ymax - endpoints.ymin) / (ylabels - 1)
	y := endpoints.ymin
	for i := range plot.Ylabel {
		plot.Ylabel[i] = fmt.Sprintf("%.2f", y)
		y += incr
	}

	// Fill in the form fields
	plot.Samples = strconv.Itoa(fs.samples)
	plot.SampleFreq = strconv.Itoa(fs.sampleFreq)
	plot.SNR = strconv.Itoa(fs.snr)
	plot.Filename = path.Base(fs.filterfile)

	showSineTable(plot)

	i := 0
	//  Fill in any previous entries so the user doesn't have to re-enter them
	for _, sig := range fs.sines {
		plot.Sines[i].Freq = strconv.Itoa(sig.freq)
		plot.Sines[i].Ampl = strconv.Itoa(int(sig.ampl))
		i++
	}

	filename := "none"
	if len(plot.Filename) > 0 {
		filename = plot.Filename
	}
	if len(plot.Status) == 0 {
		plot.Status = fmt.Sprintf("Signal consisting of %d sines was filtered with %s",
			len(fs.sines), filename)
	}

	// Write to HTTP using template and grid
	if err := filterSignalTmpl.Execute(w, plot); err != nil {
		log.Fatalf("Write to HTTP output using template with error: %v\n", err)
	}
}

// handleFilterSignal filters the signal and sends the HTML to the browser for display
func handleFilterSignal(w http.ResponseWriter, r *http.Request) {

	var plot PlotT

	// need number of samples and sample frequency to continue
	temp := r.FormValue("samples")
	if len(temp) > 0 {
		samples, err := strconv.Atoi(temp)
		if err != nil {
			plot.Status = fmt.Sprintf("Samples conversion to int error: %v", err.Error())
			showSineTable(&plot)
			fmt.Printf("Samples conversion to int error: %v\n", err.Error())
			// Write to HTTP using template and grid
			if err := filterSignalTmpl.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		temp = r.FormValue("samplefreq")
		sf, err := strconv.Atoi(temp)
		if err != nil {
			fmt.Printf("Sample frequency conversion error: %v\n", err)
			plot.Status = fmt.Sprintf("Samples frequency conversion to int error: %v", err.Error())
			showSineTable(&plot)
			// Write to HTTP using template and grid
			if err := filterSignalTmpl.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		// Get FilterState from previous submit and store in FilterSignal
		var filterState FilterState
		// This file exists only after all the samples are generated and filtered
		// Not present during block processing of the current submit
		f, err := os.Open(path.Join(dataDir, stateFile))
		if err == nil {
			defer f.Close()
			input := bufio.NewScanner(f)
			input.Scan()
			line := input.Text()
			// Last sample time from previous submit
			lst, err := strconv.ParseFloat(line, 64)
			if err != nil {
				fmt.Printf("From %s, sample time conversion error: %v\n", stateFile, err)
				plot.Status = fmt.Sprintf("Sample time conversion to int error: %v", err.Error())
				showSineTable(&plot)
				// Write to HTTP using template and grid
				if err := filterSignalTmpl.Execute(w, plot); err != nil {
					log.Fatalf("Write to HTTP output using template with error: %v\n", err)
				}
				return
			}
			filterState.lastSampleTime = lst
			filterState.lastFiltered = make([]float64, 0)
			// Get the last incomplete filtered outputs from previous submit
			for input.Scan() {
				line := input.Text()
				fltOut, err := strconv.ParseFloat(line, 64)
				if err != nil {
					fmt.Printf("Sample time conversion error: %v\n", err)
					plot.Status = fmt.Sprintf("From %s, filtered output conversion to float error: %v", stateFile, err.Error())
					showSineTable(&plot)
					// Write to HTTP using template and grid
					if err := filterSignalTmpl.Execute(w, plot); err != nil {
						log.Fatalf("Write to HTTP output using template with error: %v\n", err)
					}
					return
				}
				filterState.lastFiltered = append(filterState.lastFiltered, fltOut)
			}
		} else {
			filterState = FilterState{lastSampleTime: 0.0, lastFiltered: make([]float64, 0)}
		}

		// create FilterSignal instance fs
		fs := FilterSignal{
			sema1:            make(chan int),
			sema2:            make(chan int),
			buf:              make([][]float64, 2),
			done:             make(chan struct{}),
			samples:          samples,
			samplesGenerated: 0,
			sampleFreq:       sf,
			sines:            make([]Sine, 0),
			FilterState:      filterState,
			filterCoeff:      make([]float64, 0),
		}
		fs.buf[0] = make([]float64, block)
		fs.buf[1] = make([]float64, block)

		// start generating samples and send to filter
		err = fs.generate(r)
		if err != nil {
			plot.Status = err.Error()
			showSineTable(&plot)
			fmt.Printf("generate error: %v\n", err.Error())
			// Write to HTTP using template and grid
			if err := filterSignalTmpl.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		// start filtering when the samples arrive from the generator
		err = fs.filter(r, &plot)
		if err != nil {
			plot.Status = err.Error()
			showSineTable(&plot)
			fmt.Printf("filter error: %v\n", err.Error())
			// Write to HTTP using template and grid
			if err := filterSignalTmpl.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		// wait for the generator and filter goroutines to complete
		fs.wg.Wait()

		if err != nil {
			plot.Status = err.Error()
		}

		// Fill in the PlotT grid with the signal time vs amplitude
		err = fs.gridFillInterp(&plot)
		if err != nil {
			plot.Status = err.Error()
			showSineTable(&plot)
			fmt.Printf("gridFillInterp error: %v\n", err.Error())
			// Write to HTTP using template and grid
			if err := filterSignalTmpl.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		//  generate x-labels, ylabels, status in PlotT and execute the data on the HTML template
		fs.labelExec(w, &plot, &fs.Endpoints)

	} else {
		// delete previous state file if initial connection (not a submit)
		if err := os.Remove(path.Join(dataDir, stateFile)); err != nil {
			// ignore error if file is not present
			fmt.Printf("Remove file %s error: %v\n", path.Join(dataDir, stateFile), err)
		}
		plot.Status = "Enter samples, sample frequency, SNR, frequencies and amplitudes"
		showSineTable(&plot)
		if err := filterSignalTmpl.Execute(w, plot); err != nil {
			log.Fatalf("Write to HTTP output using template with error: %v\n", err)
		}

	}
}

// executive program
func main() {
	// Setup http server with handler for generating and filtering noisy signals
	http.HandleFunc(patternFilterSignal, handleFilterSignal)

	fmt.Printf("Filter Signal Server listening on %v.\n", addr)

	http.ListenAndServe(addr, nil)
}
