package runner

import (
	"bytes"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/x509"
	"encoding/base64"
	"encoding/pem"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"math/big"
	"net"
	"os"
	"os/exec"
	"path"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
)

var (
	useValgrind     = flag.Bool("valgrind", false, "If true, run code under valgrind")
	useGDB          = flag.Bool("gdb", false, "If true, run BoringSSL code under gdb")
	flagDebug       = flag.Bool("debug", false, "Hexdump the contents of the connection")
	mallocTest      = flag.Int64("malloc-test", -1, "If non-negative, run each test with each malloc in turn failing from the given number onwards.")
	mallocTestDebug = flag.Bool("malloc-test-debug", false, "If true, ask bssl_shim to abort rather than fail a malloc. This can be used with a specific value for --malloc-test to identity the malloc failing that is causing problems.")
	jsonOutput      = flag.String("json-output", "", "The file to output JSON results to.")
	pipe            = flag.Bool("pipe", false, "If true, print status output suitable for piping into another program.")
	testToRun       = flag.String("test", "", "The name of a test to run, or empty to run all tests")
	numWorkers      = flag.Int("num-workers", runtime.NumCPU(), "The number of workers to run in parallel.")
	shimPath        = flag.String("shim-path", "../../../build/ssl/test/bssl_shim", "The location of the shim binary.")
	resourceDir     = flag.String("resource-dir", ".", "The directory in which to find certificate and key files.")
)

const (
	rsaCertificateFile   = "cert.pem"
	ecdsaCertificateFile = "ecdsa_cert.pem"
)

const (
	rsaKeyFile       = "key.pem"
	ecdsaKeyFile     = "ecdsa_key.pem"
	channelIDKeyFile = "channel_id_key.pem"
)

var rsaCertificate, ecdsaCertificate Certificate
var channelIDKey *ecdsa.PrivateKey
var channelIDBytes []byte

var testOCSPResponse = []byte{1, 2, 3, 4}
var testSCTList = []byte{5, 6, 7, 8}

func initCertificates() {
	var err error
	rsaCertificate, err = LoadX509KeyPair(path.Join(*resourceDir, rsaCertificateFile), path.Join(*resourceDir, rsaKeyFile))
	if err != nil {
		panic(err)
	}
	rsaCertificate.OCSPStaple = testOCSPResponse
	rsaCertificate.SignedCertificateTimestampList = testSCTList

	ecdsaCertificate, err = LoadX509KeyPair(path.Join(*resourceDir, ecdsaCertificateFile), path.Join(*resourceDir, ecdsaKeyFile))
	if err != nil {
		panic(err)
	}
	ecdsaCertificate.OCSPStaple = testOCSPResponse
	ecdsaCertificate.SignedCertificateTimestampList = testSCTList

	channelIDPEMBlock, err := ioutil.ReadFile(path.Join(*resourceDir, channelIDKeyFile))
	if err != nil {
		panic(err)
	}
	channelIDDERBlock, _ := pem.Decode(channelIDPEMBlock)
	if channelIDDERBlock.Type != "EC PRIVATE KEY" {
		panic("bad key type")
	}
	channelIDKey, err = x509.ParseECPrivateKey(channelIDDERBlock.Bytes)
	if err != nil {
		panic(err)
	}
	if channelIDKey.Curve != elliptic.P256() {
		panic("bad curve")
	}

	channelIDBytes = make([]byte, 64)
	writeIntPadded(channelIDBytes[:32], channelIDKey.X)
	writeIntPadded(channelIDBytes[32:], channelIDKey.Y)
}

var certificateOnce sync.Once

func getRSACertificate() Certificate {
	certificateOnce.Do(initCertificates)
	return rsaCertificate
}

func getECDSACertificate() Certificate {
	certificateOnce.Do(initCertificates)
	return ecdsaCertificate
}

type testType int

const (
	clientTest testType = iota
	serverTest
)

type protocol int

const (
	tls protocol = iota
	dtls
)

const (
	alpn = 1
	npn  = 2
)

type testCase struct {
	testType      testType
	protocol      protocol
	name          string
	config        Config
	shouldFail    bool
	expectedError string
	// expectedLocalError, if not empty, contains a substring that must be
	// found in the local error.
	expectedLocalError string
	// expectedVersion, if non-zero, specifies the TLS version that must be
	// negotiated.
	expectedVersion uint16
	// expectedResumeVersion, if non-zero, specifies the TLS version that
	// must be negotiated on resumption. If zero, expectedVersion is used.
	expectedResumeVersion uint16
	// expectedCipher, if non-zero, specifies the TLS cipher suite that
	// should be negotiated.
	expectedCipher uint16
	// expectChannelID controls whether the connection should have
	// negotiated a Channel ID with channelIDKey.
	expectChannelID bool
	// expectedNextProto controls whether the connection should
	// negotiate a next protocol via NPN or ALPN.
	expectedNextProto string
	// expectedNextProtoType, if non-zero, is the expected next
	// protocol negotiation mechanism.
	expectedNextProtoType int
	// expectedSRTPProtectionProfile is the DTLS-SRTP profile that
	// should be negotiated. If zero, none should be negotiated.
	expectedSRTPProtectionProfile uint16
	// expectedOCSPResponse, if not nil, is the expected OCSP response to be received.
	expectedOCSPResponse []uint8
	// expectedSCTList, if not nil, is the expected SCT list to be received.
	expectedSCTList []uint8
	// expectedClientCertSignatureHash, if not zero, is the TLS id of the
	// hash function that the client should have used when signing the
	// handshake with a client certificate.
	expectedClientCertSignatureHash uint8
	// messageLen is the length, in bytes, of the test message that will be
	// sent.
	messageLen int
	// messageCount is the number of test messages that will be sent.
	messageCount int
	// digestPrefs is the list of digest preferences from the client.
	digestPrefs string
	// certFile is the path to the certificate to use for the server.
	certFile string
	// keyFile is the path to the private key to use for the server.
	keyFile string
	// resumeSession controls whether a second connection should be tested
	// which attempts to resume the first session.
	resumeSession bool
	// expectResumeRejected, if true, specifies that the attempted
	// resumption must be rejected by the client. This is only valid for a
	// serverTest.
	expectResumeRejected bool
	// resumeConfig, if not nil, points to a Config to be used on
	// resumption. Unless newSessionsOnResume is set,
	// SessionTicketKey, ServerSessionCache, and
	// ClientSessionCache are copied from the initial connection's
	// config. If nil, the initial connection's config is used.
	resumeConfig *Config
	// newSessionsOnResume, if true, will cause resumeConfig to
	// use a different session resumption context.
	newSessionsOnResume bool
	// noSessionCache, if true, will cause the server to run without a
	// session cache.
	noSessionCache bool
	// sendPrefix sends a prefix on the socket before actually performing a
	// handshake.
	sendPrefix string
	// shimWritesFirst controls whether the shim sends an initial "hello"
	// message before doing a roundtrip with the runner.
	shimWritesFirst bool
	// shimShutsDown, if true, runs a test where the shim shuts down the
	// connection immediately after the handshake rather than echoing
	// messages from the runner.
	shimShutsDown bool
	// renegotiate indicates the the connection should be renegotiated
	// during the exchange.
	renegotiate bool
	// renegotiateCiphers is a list of ciphersuite ids that will be
	// switched in just before renegotiation.
	renegotiateCiphers []uint16
	// replayWrites, if true, configures the underlying transport
	// to replay every write it makes in DTLS tests.
	replayWrites bool
	// damageFirstWrite, if true, configures the underlying transport to
	// damage the final byte of the first application data write.
	damageFirstWrite bool
	// exportKeyingMaterial, if non-zero, configures the test to exchange
	// keying material and verify they match.
	exportKeyingMaterial int
	exportLabel          string
	exportContext        string
	useExportContext     bool
	// flags, if not empty, contains a list of command-line flags that will
	// be passed to the shim program.
	flags []string
	// testTLSUnique, if true, causes the shim to send the tls-unique value
	// which will be compared against the expected value.
	testTLSUnique bool
	// sendEmptyRecords is the number of consecutive empty records to send
	// before and after the test message.
	sendEmptyRecords int
	// sendWarningAlerts is the number of consecutive warning alerts to send
	// before and after the test message.
	sendWarningAlerts int
	// expectMessageDropped, if true, means the test message is expected to
	// be dropped by the client rather than echoed back.
	expectMessageDropped bool
}

var testCases []testCase

func doExchange(test *testCase, config *Config, conn net.Conn, isResume bool) error {
	var connDebug *recordingConn
	var connDamage *damageAdaptor
	if *flagDebug {
		connDebug = &recordingConn{Conn: conn}
		conn = connDebug
		defer func() {
			connDebug.WriteTo(os.Stdout)
		}()
	}

	if test.protocol == dtls {
		config.Bugs.PacketAdaptor = newPacketAdaptor(conn)
		conn = config.Bugs.PacketAdaptor
		if test.replayWrites {
			conn = newReplayAdaptor(conn)
		}
	}

	if test.damageFirstWrite {
		connDamage = newDamageAdaptor(conn)
		conn = connDamage
	}

	if test.sendPrefix != "" {
		if _, err := conn.Write([]byte(test.sendPrefix)); err != nil {
			return err
		}
	}

	var tlsConn *Conn
	if test.testType == clientTest {
		if test.protocol == dtls {
			tlsConn = DTLSServer(conn, config)
		} else {
			tlsConn = Server(conn, config)
		}
	} else {
		config.InsecureSkipVerify = true
		if test.protocol == dtls {
			tlsConn = DTLSClient(conn, config)
		} else {
			tlsConn = Client(conn, config)
		}
	}
	defer tlsConn.Close()

	if err := tlsConn.Handshake(); err != nil {
		return err
	}

	// TODO(davidben): move all per-connection expectations into a dedicated
	// expectations struct that can be specified separately for the two
	// legs.
	expectedVersion := test.expectedVersion
	if isResume && test.expectedResumeVersion != 0 {
		expectedVersion = test.expectedResumeVersion
	}
	connState := tlsConn.ConnectionState()
	if vers := connState.Version; expectedVersion != 0 && vers != expectedVersion {
		return fmt.Errorf("got version %x, expected %x", vers, expectedVersion)
	}

	if cipher := connState.CipherSuite; test.expectedCipher != 0 && cipher != test.expectedCipher {
		return fmt.Errorf("got cipher %x, expected %x", cipher, test.expectedCipher)
	}
	if didResume := connState.DidResume; isResume && didResume == test.expectResumeRejected {
		return fmt.Errorf("didResume is %t, but we expected the opposite", didResume)
	}

	if test.expectChannelID {
		channelID := connState.ChannelID
		if channelID == nil {
			return fmt.Errorf("no channel ID negotiated")
		}
		if channelID.Curve != channelIDKey.Curve ||
			channelIDKey.X.Cmp(channelIDKey.X) != 0 ||
			channelIDKey.Y.Cmp(channelIDKey.Y) != 0 {
			return fmt.Errorf("incorrect channel ID")
		}
	}

	if expected := test.expectedNextProto; expected != "" {
		if actual := connState.NegotiatedProtocol; actual != expected {
			return fmt.Errorf("next proto mismatch: got %s, wanted %s", actual, expected)
		}
	}

	if test.expectedNextProtoType != 0 {
		if (test.expectedNextProtoType == alpn) != connState.NegotiatedProtocolFromALPN {
			return fmt.Errorf("next proto type mismatch")
		}
	}

	if p := connState.SRTPProtectionProfile; p != test.expectedSRTPProtectionProfile {
		return fmt.Errorf("SRTP profile mismatch: got %d, wanted %d", p, test.expectedSRTPProtectionProfile)
	}

	if test.expectedOCSPResponse != nil && !bytes.Equal(test.expectedOCSPResponse, tlsConn.OCSPResponse()) {
		return fmt.Errorf("OCSP Response mismatch")
	}

	if test.expectedSCTList != nil && !bytes.Equal(test.expectedSCTList, connState.SCTList) {
		return fmt.Errorf("SCT list mismatch")
	}

	if expected := test.expectedClientCertSignatureHash; expected != 0 && expected != connState.ClientCertSignatureHash {
		return fmt.Errorf("expected client to sign handshake with hash %d, but got %d", expected, connState.ClientCertSignatureHash)
	}

	if test.exportKeyingMaterial > 0 {
		actual := make([]byte, test.exportKeyingMaterial)
		if _, err := io.ReadFull(tlsConn, actual); err != nil {
			return err
		}
		expected, err := tlsConn.ExportKeyingMaterial(test.exportKeyingMaterial, []byte(test.exportLabel), []byte(test.exportContext), test.useExportContext)
		if err != nil {
			return err
		}
		if !bytes.Equal(actual, expected) {
			return fmt.Errorf("keying material mismatch")
		}
	}

	if test.testTLSUnique {
		var peersValue [12]byte
		if _, err := io.ReadFull(tlsConn, peersValue[:]); err != nil {
			return err
		}
		expected := tlsConn.ConnectionState().TLSUnique
		if !bytes.Equal(peersValue[:], expected) {
			return fmt.Errorf("tls-unique mismatch: peer sent %x, but %x was expected", peersValue[:], expected)
		}
	}

	if test.shimWritesFirst {
		var buf [5]byte
		_, err := io.ReadFull(tlsConn, buf[:])
		if err != nil {
			return err
		}
		if string(buf[:]) != "hello" {
			return fmt.Errorf("bad initial message")
		}
	}

	for i := 0; i < test.sendEmptyRecords; i++ {
		tlsConn.Write(nil)
	}

	for i := 0; i < test.sendWarningAlerts; i++ {
		tlsConn.SendAlert(alertLevelWarning, alertUnexpectedMessage)
	}

	if test.renegotiate {
		if test.renegotiateCiphers != nil {
			config.CipherSuites = test.renegotiateCiphers
		}
		if err := tlsConn.Renegotiate(); err != nil {
			return err
		}
	} else if test.renegotiateCiphers != nil {
		panic("renegotiateCiphers without renegotiate")
	}

	if test.damageFirstWrite {
		connDamage.setDamage(true)
		tlsConn.Write([]byte("DAMAGED WRITE"))
		connDamage.setDamage(false)
	}

	messageLen := test.messageLen
	if messageLen < 0 {
		if test.protocol == dtls {
			return fmt.Errorf("messageLen < 0 not supported for DTLS tests")
		}
		// Read until EOF.
		_, err := io.Copy(ioutil.Discard, tlsConn)
		return err
	}
	if messageLen == 0 {
		messageLen = 32
	}

	messageCount := test.messageCount
	if messageCount == 0 {
		messageCount = 1
	}

	for j := 0; j < messageCount; j++ {
		testMessage := make([]byte, messageLen)
		for i := range testMessage {
			testMessage[i] = 0x42 ^ byte(j)
		}
		tlsConn.Write(testMessage)

		for i := 0; i < test.sendEmptyRecords; i++ {
			tlsConn.Write(nil)
		}

		for i := 0; i < test.sendWarningAlerts; i++ {
			tlsConn.SendAlert(alertLevelWarning, alertUnexpectedMessage)
		}

		if test.shimShutsDown || test.expectMessageDropped {
			// The shim will not respond.
			continue
		}

		buf := make([]byte, len(testMessage))
		if test.protocol == dtls {
			bufTmp := make([]byte, len(buf)+1)
			n, err := tlsConn.Read(bufTmp)
			if err != nil {
				return err
			}
			if n != len(buf) {
				return fmt.Errorf("bad reply; length mismatch (%d vs %d)", n, len(buf))
			}
			copy(buf, bufTmp)
		} else {
			_, err := io.ReadFull(tlsConn, buf)
			if err != nil {
				return err
			}
		}

		for i, v := range buf {
			if v != testMessage[i]^0xff {
				return fmt.Errorf("bad reply contents at byte %d", i)
			}
		}
	}

	return nil
}

func valgrindOf(dbAttach bool, path string, args ...string) *exec.Cmd {
	valgrindArgs := []string{"--error-exitcode=99", "--track-origins=yes", "--leak-check=full"}
	if dbAttach {
		valgrindArgs = append(valgrindArgs, "--db-attach=yes", "--db-command=xterm -e gdb -nw %f %p")
	}
	valgrindArgs = append(valgrindArgs, path)
	valgrindArgs = append(valgrindArgs, args...)

	return exec.Command("valgrind", valgrindArgs...)
}

func gdbOf(path string, args ...string) *exec.Cmd {
	xtermArgs := []string{"-e", "gdb", "--args"}
	xtermArgs = append(xtermArgs, path)
	xtermArgs = append(xtermArgs, args...)

	return exec.Command("xterm", xtermArgs...)
}

type moreMallocsError struct{}

func (moreMallocsError) Error() string {
	return "child process did not exhaust all allocation calls"
}

var errMoreMallocs = moreMallocsError{}

// accept accepts a connection from listener, unless waitChan signals a process
// exit first.
func acceptOrWait(listener net.Listener, waitChan chan error) (net.Conn, error) {
	type connOrError struct {
		conn net.Conn
		err  error
	}
	connChan := make(chan connOrError, 1)
	go func() {
		conn, err := listener.Accept()
		connChan <- connOrError{conn, err}
		close(connChan)
	}()
	select {
	case result := <-connChan:
		return result.conn, result.err
	case childErr := <-waitChan:
		waitChan <- childErr
		return nil, fmt.Errorf("child exited early: %s", childErr)
	}
}

func runTest(test *testCase, shimPath string, mallocNumToFail int64) error {
	if !test.shouldFail && (len(test.expectedError) > 0 || len(test.expectedLocalError) > 0) {
		panic("Error expected without shouldFail in " + test.name)
	}

	if test.expectResumeRejected && !test.resumeSession {
		panic("expectResumeRejected without resumeSession in " + test.name)
	}

	if test.testType != clientTest && test.expectedClientCertSignatureHash != 0 {
		panic("expectedClientCertSignatureHash non-zero with serverTest in " + test.name)
	}

	listener, err := net.ListenTCP("tcp4", &net.TCPAddr{IP: net.IP{127, 0, 0, 1}})
	if err != nil {
		panic(err)
	}
	defer func() {
		if listener != nil {
			listener.Close()
		}
	}()

	flags := []string{"-port", strconv.Itoa(listener.Addr().(*net.TCPAddr).Port)}
	if test.testType == serverTest {
		flags = append(flags, "-server")

		flags = append(flags, "-key-file")
		if test.keyFile == "" {
			flags = append(flags, path.Join(*resourceDir, rsaKeyFile))
		} else {
			flags = append(flags, path.Join(*resourceDir, test.keyFile))
		}

		flags = append(flags, "-cert-file")
		if test.certFile == "" {
			flags = append(flags, path.Join(*resourceDir, rsaCertificateFile))
		} else {
			flags = append(flags, path.Join(*resourceDir, test.certFile))
		}
	}

	if test.digestPrefs != "" {
		flags = append(flags, "-digest-prefs")
		flags = append(flags, test.digestPrefs)
	}

	if test.protocol == dtls {
		flags = append(flags, "-dtls")
	}

	if test.resumeSession {
		flags = append(flags, "-resume")
	}

	if test.shimWritesFirst {
		flags = append(flags, "-shim-writes-first")
	}

	if test.shimShutsDown {
		flags = append(flags, "-shim-shuts-down")
	}

	if test.exportKeyingMaterial > 0 {
		flags = append(flags, "-export-keying-material", strconv.Itoa(test.exportKeyingMaterial))
		flags = append(flags, "-export-label", test.exportLabel)
		flags = append(flags, "-export-context", test.exportContext)
		if test.useExportContext {
			flags = append(flags, "-use-export-context")
		}
	}
	if test.expectResumeRejected {
		flags = append(flags, "-expect-session-miss")
	}

	if test.testTLSUnique {
		flags = append(flags, "-tls-unique")
	}

	flags = append(flags, test.flags...)

	var shim *exec.Cmd
	if *useValgrind {
		shim = valgrindOf(false, shimPath, flags...)
	} else if *useGDB {
		shim = gdbOf(shimPath, flags...)
	} else {
		shim = exec.Command(shimPath, flags...)
	}
	shim.Stdin = os.Stdin
	var stdoutBuf, stderrBuf bytes.Buffer
	shim.Stdout = &stdoutBuf
	shim.Stderr = &stderrBuf
	if mallocNumToFail >= 0 {
		shim.Env = os.Environ()
		shim.Env = append(shim.Env, "MALLOC_NUMBER_TO_FAIL="+strconv.FormatInt(mallocNumToFail, 10))
		if *mallocTestDebug {
			shim.Env = append(shim.Env, "MALLOC_BREAK_ON_FAIL=1")
		}
		shim.Env = append(shim.Env, "_MALLOC_CHECK=1")
	}

	if err := shim.Start(); err != nil {
		panic(err)
	}
	waitChan := make(chan error, 1)
	go func() { waitChan <- shim.Wait() }()

	config := test.config
	if !test.noSessionCache {
		config.ClientSessionCache = NewLRUClientSessionCache(1)
		config.ServerSessionCache = NewLRUServerSessionCache(1)
	}
	if test.testType == clientTest {
		if len(config.Certificates) == 0 {
			config.Certificates = []Certificate{getRSACertificate()}
		}
	} else {
		// Supply a ServerName to ensure a constant session cache key,
		// rather than falling back to net.Conn.RemoteAddr.
		if len(config.ServerName) == 0 {
			config.ServerName = "test"
		}
	}

	conn, err := acceptOrWait(listener, waitChan)
	if err == nil {
		err = doExchange(test, &config, conn, false /* not a resumption */)
		conn.Close()
	}

	if err == nil && test.resumeSession {
		var resumeConfig Config
		if test.resumeConfig != nil {
			resumeConfig = *test.resumeConfig
			if len(resumeConfig.ServerName) == 0 {
				resumeConfig.ServerName = config.ServerName
			}
			if len(resumeConfig.Certificates) == 0 {
				resumeConfig.Certificates = []Certificate{getRSACertificate()}
			}
			if test.newSessionsOnResume {
				if !test.noSessionCache {
					resumeConfig.ClientSessionCache = NewLRUClientSessionCache(1)
					resumeConfig.ServerSessionCache = NewLRUServerSessionCache(1)
				}
			} else {
				resumeConfig.SessionTicketKey = config.SessionTicketKey
				resumeConfig.ClientSessionCache = config.ClientSessionCache
				resumeConfig.ServerSessionCache = config.ServerSessionCache
			}
		} else {
			resumeConfig = config
		}
		var connResume net.Conn
		connResume, err = acceptOrWait(listener, waitChan)
		if err == nil {
			err = doExchange(test, &resumeConfig, connResume, true /* resumption */)
			connResume.Close()
		}
	}

	// Close the listener now. This is to avoid hangs should the shim try to
	// open more connections than expected.
	listener.Close()
	listener = nil

	childErr := <-waitChan
	if exitError, ok := childErr.(*exec.ExitError); ok {
		if exitError.Sys().(syscall.WaitStatus).ExitStatus() == 88 {
			return errMoreMallocs
		}
	}

	stdout := string(stdoutBuf.Bytes())
	stderr := string(stderrBuf.Bytes())
	failed := err != nil || childErr != nil
	correctFailure := len(test.expectedError) == 0 || strings.Contains(stderr, test.expectedError)
	localError := "none"
	if err != nil {
		localError = err.Error()
	}
	if len(test.expectedLocalError) != 0 {
		correctFailure = correctFailure && strings.Contains(localError, test.expectedLocalError)
	}

	if failed != test.shouldFail || failed && !correctFailure {
		childError := "none"
		if childErr != nil {
			childError = childErr.Error()
		}

		var msg string
		switch {
		case failed && !test.shouldFail:
			msg = "unexpected failure"
		case !failed && test.shouldFail:
			msg = "unexpected success"
		case failed && !correctFailure:
			msg = "bad error (wanted '" + test.expectedError + "' / '" + test.expectedLocalError + "')"
		default:
			panic("internal error")
		}

		return fmt.Errorf("%s: local error '%s', child error '%s', stdout:\n%s\nstderr:\n%s", msg, localError, childError, stdout, stderr)
	}

	if !*useValgrind && !failed && len(stderr) > 0 {
		println(stderr)
	}

	return nil
}

var tlsVersions = []struct {
	name    string
	version uint16
	flag    string
	hasDTLS bool
}{
	{"SSL3", VersionSSL30, "-no-ssl3", false},
	{"TLS1", VersionTLS10, "-no-tls1", true},
	{"TLS11", VersionTLS11, "-no-tls11", false},
	{"TLS12", VersionTLS12, "-no-tls12", true},
}

var testCipherSuites = []struct {
	name string
	id   uint16
}{
	{"3DES-SHA", TLS_RSA_WITH_3DES_EDE_CBC_SHA},
	{"AES128-GCM", TLS_RSA_WITH_AES_128_GCM_SHA256},
	{"AES128-SHA", TLS_RSA_WITH_AES_128_CBC_SHA},
	{"AES128-SHA256", TLS_RSA_WITH_AES_128_CBC_SHA256},
	{"AES256-GCM", TLS_RSA_WITH_AES_256_GCM_SHA384},
	{"AES256-SHA", TLS_RSA_WITH_AES_256_CBC_SHA},
	{"AES256-SHA256", TLS_RSA_WITH_AES_256_CBC_SHA256},
	{"DHE-RSA-AES128-GCM", TLS_DHE_RSA_WITH_AES_128_GCM_SHA256},
	{"DHE-RSA-AES128-SHA", TLS_DHE_RSA_WITH_AES_128_CBC_SHA},
	{"DHE-RSA-AES128-SHA256", TLS_DHE_RSA_WITH_AES_128_CBC_SHA256},
	{"DHE-RSA-AES256-GCM", TLS_DHE_RSA_WITH_AES_256_GCM_SHA384},
	{"DHE-RSA-AES256-SHA", TLS_DHE_RSA_WITH_AES_256_CBC_SHA},
	{"DHE-RSA-AES256-SHA256", TLS_DHE_RSA_WITH_AES_256_CBC_SHA256},
	{"ECDHE-ECDSA-AES128-GCM", TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256},
	{"ECDHE-ECDSA-AES128-SHA", TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA},
	{"ECDHE-ECDSA-AES128-SHA256", TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA256},
	{"ECDHE-ECDSA-AES256-GCM", TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384},
	{"ECDHE-ECDSA-AES256-SHA", TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA},
	{"ECDHE-ECDSA-AES256-SHA384", TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA384},
	{"ECDHE-ECDSA-CHACHA20-POLY1305", TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256},
	{"ECDHE-ECDSA-RC4-SHA", TLS_ECDHE_ECDSA_WITH_RC4_128_SHA},
	{"ECDHE-RSA-AES128-GCM", TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256},
	{"ECDHE-RSA-AES128-SHA", TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA},
	{"ECDHE-RSA-AES128-SHA256", TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA256},
	{"ECDHE-RSA-AES256-GCM", TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384},
	{"ECDHE-RSA-AES256-SHA", TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA},
	{"ECDHE-RSA-AES256-SHA384", TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA384},
	{"ECDHE-RSA-CHACHA20-POLY1305", TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256},
	{"ECDHE-RSA-RC4-SHA", TLS_ECDHE_RSA_WITH_RC4_128_SHA},
	{"PSK-AES128-CBC-SHA", TLS_PSK_WITH_AES_128_CBC_SHA},
	{"PSK-AES256-CBC-SHA", TLS_PSK_WITH_AES_256_CBC_SHA},
	{"ECDHE-PSK-AES128-CBC-SHA", TLS_ECDHE_PSK_WITH_AES_128_CBC_SHA},
	{"ECDHE-PSK-AES256-CBC-SHA", TLS_ECDHE_PSK_WITH_AES_256_CBC_SHA},
	{"PSK-RC4-SHA", TLS_PSK_WITH_RC4_128_SHA},
	{"RC4-MD5", TLS_RSA_WITH_RC4_128_MD5},
	{"RC4-SHA", TLS_RSA_WITH_RC4_128_SHA},
	{"NULL-SHA", TLS_RSA_WITH_NULL_SHA},
}

func hasComponent(suiteName, component string) bool {
	return strings.Contains("-"+suiteName+"-", "-"+component+"-")
}

func isTLS12Only(suiteName string) bool {
	return hasComponent(suiteName, "GCM") ||
		hasComponent(suiteName, "SHA256") ||
		hasComponent(suiteName, "SHA384") ||
		hasComponent(suiteName, "POLY1305")
}

func isDTLSCipher(suiteName string) bool {
	return !hasComponent(suiteName, "RC4") && !hasComponent(suiteName, "NULL")
}

func bigFromHex(hex string) *big.Int {
	ret, ok := new(big.Int).SetString(hex, 16)
	if !ok {
		panic("failed to parse hex number 0x" + hex)
	}
	return ret
}

func addBasicTests() {
	basicTests := []testCase{
		{
			name: "BadRSASignature",
			config: Config{
				CipherSuites: []uint16{TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256},
				Bugs: ProtocolBugs{
					InvalidSKXSignature: true,
				},
			},
			shouldFail:    true,
			expectedError: ":BAD_SIGNATURE:",
		},
		{
			name: "BadECDSASignature",
			config: Config{
				CipherSuites: []uint16{TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256},
				Bugs: ProtocolBugs{
					InvalidSKXSignature: true,
				},
				Certificates: []Certificate{getECDSACertificate()},
			},
			shouldFail:    true,
			expectedError: ":BAD_SIGNATURE:",
		},
		{
			testType: serverTest,
			name:     "BadRSASignature-ClientAuth",
			config: Config{
				Bugs: ProtocolBugs{
					InvalidCertVerifySignature: true,
				},
				Certificates: []Certificate{getRSACertificate()},
			},
			shouldFail:    true,
			expectedError: ":BAD_SIGNATURE:",
			flags:         []string{"-require-any-client-certificate"},
		},
		{
			testType: serverTest,
			name:     "BadECDSASignature-ClientAuth",
			config: Config{
				Bugs: ProtocolBugs{
					InvalidCertVerifySignature: true,
				},
				Certificates: []Certificate{getECDSACertificate()},
			},
			shouldFail:    true,
			expectedError: ":BAD_SIGNATURE:",
			flags:         []string{"-require-any-client-certificate"},
		},
		{
			name: "BadECDSACurve",
			config: Config{
				CipherSuites: []uint16{TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256},
				Bugs: ProtocolBugs{
					InvalidSKXCurve: true,
				},
				Certificates: []Certificate{getECDSACertificate()},
			},
			shouldFail:    true,
			expectedError: ":WRONG_CURVE:",
		},
		{
			testType: serverTest,
			name:     "BadRSAVersion",
			config: Config{
				CipherSuites: []uint16{TLS_RSA_WITH_RC4_128_SHA},
				Bugs: ProtocolBugs{
					RsaClientKeyExchangeVersion: VersionTLS11,
				},
			},
			shouldFail:    true,
			expectedError: ":DECRYPTION_FAILED_OR_BAD_RECORD_MAC:",
		},
		{
			name: "NoFallbackSCSV",
			config: Config{
				Bugs: ProtocolBugs{
					FailIfNotFallbackSCSV: true,
				},
			},
			shouldFail:         true,
			expectedLocalError: "no fallback SCSV found",
		},
		{
			name: "SendFallbackSCSV",
			config: Config{
				Bugs: ProtocolBugs{
					FailIfNotFallbackSCSV: true,
				},
			},
			flags: []string{"-fallback-scsv"},
		},
		{
			name: "ClientCertificateTypes",
			config: Config{
				ClientAuth: RequestClientCert,
				ClientCertificateTypes: []byte{
					CertTypeDSSSign,
					CertTypeRSASign,
					CertTypeECDSASign,
				},
			},
			flags: []string{
				"-expect-certificate-types",
				base64.StdEncoding.EncodeToString([]byte{
					CertTypeDSSSign,
					CertTypeRSASign,
					CertTypeECDSASign,
				}),
			},
		},
		{
			name: "NoClientCertificate",
			config: Config{
				ClientAuth: RequireAnyClientCert,
			},
			shouldFail:         true,
			expectedLocalError: "client didn't provide a certificate",
		},
		{
			name: "UnauthenticatedECDH",
			config: Config{
				CipherSuites: []uint16{TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256},
				Bugs: ProtocolBugs{
					UnauthenticatedECDH: true,
				},
			},
			shouldFail:    true,
			expectedError: ":UNEXPECTED_MESSAGE:",
		},
		{
			name: "SkipCertificateStatus",
			config: Config{
				CipherSuites: []uint16{TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256},
				Bugs: ProtocolBugs{
					SkipCertificateStatus: true,
				},
			},
			flags: []string{
				"-enable-ocsp-stapling",
			},
		},
		{
			name: "SkipServerKeyExchange",
			config: Config{
				CipherSuites: []uint16{TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256},
				Bugs: ProtocolBugs{
					SkipServerKeyExchange: true,
				},
			},
			shouldFail:    true,
			expectedError: ":UNEXPECTED_MESSAGE:",
		},
		{
			name: "SkipChangeCipherSpec-Client",
			config: Config{
				Bugs: ProtocolBugs{
					SkipChangeCipherSpec: true,
				},
			},
			shouldFail:    true,
			expectedError: ":HANDSHAKE_RECORD_BEFORE_CCS:",
		},
		{
			testType: serverTest,
			name:     "SkipChangeCipherSpec-Server",
			config: Config{
				Bugs: ProtocolBugs{
					SkipChangeCipherSpec: true,
				},
			},
			shouldFail:    true,
			expectedError: ":HANDSHAKE_RECORD_BEFORE_CCS:",
		},
		{
			testType: serverTest,
			name:     "SkipChangeCipherSpec-Server-NPN",
			config: Config{
				NextProtos: []string{"bar"},
				Bugs: ProtocolBugs{
					SkipChangeCipherSpec: true,
				},
			},
			flags: []string{
				"-advertise-npn", "\x03foo\x03bar\x03baz",
			},
			shouldFail:    true,
			expectedError: ":HANDSHAKE_RECORD_BEFORE_CCS:",
		},
		{
			name: "FragmentAcrossChangeCipherSpec-Client",
			config: Config{
				Bugs: ProtocolBugs{
					FragmentAcrossChangeCipherSpec: true,
				},
			},
			shouldFail:    true,
			expectedError: ":HANDSHAKE_RECORD_BEFORE_CCS:",
		},
		{
			testType: serverTest,
			name:     "FragmentAcrossChangeCipherSpec-Server",
			config: Config{
				Bugs: ProtocolBugs{
					FragmentAcrossChangeCipherSpec: true,
				},
			},
			shouldFail:    true,
			expectedError: ":HANDSHAKE_RECORD_BEFORE_CCS:",
		},
		{
			testType: serverTest,
			name:     "FragmentAcrossChangeCipherSpec-Server-NPN",
			config: Config{
				NextProtos: []string{"bar"},
				Bugs: ProtocolBugs{
					FragmentAcrossChangeCipherSpec: true,
				},
			},
			flags: []string{
				"-advertise-npn", "\x03foo\x03bar\x03baz",
			},
			shouldFail:    true,
			expectedError: ":HANDSHAKE_RECORD_BEFORE_CCS:",
		},
		{
			testType: serverTest,
			name:     "Alert",
			config: Config{
				Bugs: ProtocolBugs{
					SendSpuriousAlert: alertRecordOverflow,
				},
			},
			shouldFail:    true,
			expectedError: ":TLSV1_ALERT_RECORD_OVERFLOW:",
		},
		{
			protocol: dtls,
			testType: serverTest,
			name:     "Alert-DTLS",
			config: Config{
				Bugs: ProtocolBugs{
					SendSpuriousAlert: alertRecordOverflow,
				},
			},
			shouldFail:    true,
			expectedError: ":TLSV1_ALERT_RECORD_OVERFLOW:",
		},
		{
			testType: serverTest,
			name:     "FragmentAlert",
			config: Config{
				Bugs: ProtocolBugs{
					FragmentAlert:     true,
					SendSpuriousAlert: alertRecordOverflow,
				},
			},
			shouldFail:    true,
			expectedError: ":BAD_ALERT:",
		},
		{
			protocol: dtls,
			testType: serverTest,
			name:     "FragmentAlert-DTLS",
			config: Config{
				Bugs: ProtocolBugs{
					FragmentAlert:     true,
					SendSpuriousAlert: alertRecordOverflow,
				},
			},
			shouldFail:    true,
			expectedError: ":BAD_ALERT:",
		},
		{
			testType: serverTest,
			name:     "EarlyChangeCipherSpec-server-1",
			config: Config{
				Bugs: ProtocolBugs{
					EarlyChangeCipherSpec: 1,
				},
			},
			shouldFail:    true,
			expectedError: ":CCS_RECEIVED_EARLY:",
		},
		{
			testType: serverTest,
			name:     "EarlyChangeCipherSpec-server-2",
			config: Config{
				Bugs: ProtocolBugs{
					EarlyChangeCipherSpec: 2,
				},
			},
			shouldFail:    true,
			expectedError: ":CCS_RECEIVED_EARLY:",
		},
		{
			name: "SkipNewSessionTicket",
			config: Config{
				Bugs: ProtocolBugs{
					SkipNewSessionTicket: true,
				},
			},
			shouldFail:    true,
			expectedError: ":CCS_RECEIVED_EARLY:",
		},
		{
			testType: serverTest,
			name:     "FallbackSCSV",
			config: Config{
				MaxVersion: VersionTLS11,
				Bugs: ProtocolBugs{
					SendFallbackSCSV: true,
				},
			},
			shouldFail:    true,
			expectedError: ":INAPPROPRIATE_FALLBACK:",
		},
		{
			testType: serverTest,
			name:     "FallbackSCSV-VersionMatch",
			config: Config{
				Bugs: ProtocolBugs{
					SendFallbackSCSV: true,
				},
			},
		},
		{
			testType: serverTest,
			name:     "FragmentedClientVersion",
			config: Config{
				Bugs: ProtocolBugs{
					MaxHandshakeRecordLength: 1,
					FragmentClientVersion:    true,
				},
			},
			expectedVersion: VersionTLS12,
		},
		{
			testType: serverTest,
			name:     "MinorVersionTolerance",
			config: Config{
				Bugs: ProtocolBugs{
					SendClientVersion: 0x03ff,
				},
			},
			expectedVersion: VersionTLS12,
		},
		{
			testType: serverTest,
			name:     "MajorVersionTolerance",
			config: Config{
				Bugs: ProtocolBugs{
					SendClientVersion: 0x0400,
				},
			},
			expectedVersion: VersionTLS12,
		},
		{
			testType: serverTest,
			name:     "VersionTooLow",
			config: Config{
				Bugs: ProtocolBugs{
					SendClientVersion: 0x0200,
				},
			},
			shouldFail:    true,
			expectedError: ":UNSUPPORTED_PROTOCOL:",
		},
		{
			testType:      serverTest,
			name:          "HttpGET",
			sendPrefix:    "GET / HTTP/1.0\n",
			shouldFail:    true,
			expectedError: ":HTTP_REQUEST:",
		},
		{
			testType:      serverTest,
			name:          "HttpPOST",
			sendPrefix:    "POST / HTTP/1.0\n",
			shouldFail:    true,
			expectedError: ":HTTP_REQUEST:",
		},
		{
			testType:      serverTest,
			name:          "HttpHEAD",
			sendPrefix:    "HEAD / HTTP/1.0\n",
			shouldFail:    true,
			expectedError: ":HTTP_REQUEST:",
		},
		{
			testType:      serverTest,
			name:          "HttpPUT",
			sendPrefix:    "PUT / HTTP/1.0\n",
			shouldFail:    true,
			expectedError: ":HTTP_REQUEST:",
		},
		{
			testType:      serverTest,
			name:          "HttpCONNECT",
			sendPrefix:    "CONNECT www.google.com:443 HTTP/1.0\n",
			shouldFail:    true,
			expectedError: ":HTTPS_PROXY_REQUEST:",
		},
		{
			testType:      serverTest,
			name:          "Garbage",
			sendPrefix:    "blah",
			shouldFail:    true,
			expectedError: ":WRONG_VERSION_NUMBER:",
		},
		{
			name: "SkipCipherVersionCheck",
			config: Config{
				CipherSuites: []uint16{TLS_RSA_WITH_AES_128_GCM_SHA256},
				MaxVersion:   VersionTLS11,
				Bugs: ProtocolBugs{
					SkipCipherVersionCheck: true,
				},
			},
			shouldFail:    true,
			expectedError: ":WRONG_CIPHER_RETURNED:",
		},
		{
			name: "RSAEphemeralKey",
			config: Config{
				CipherSuites: []uint16{TLS_RSA_WITH_AES_128_CBC_SHA},
				Bugs: ProtocolBugs{
					RSAEphemeralKey: true,
				},
			},
			shouldFail:    true,
			expectedError: ":UNEXPECTED_MESSAGE:",
		},
		{
			name:          "DisableEverything",
			flags:         []string{"-no-tls12", "-no-tls11", "-no-tls1", "-no-ssl3"},
			shouldFail:    true,
			expectedError: ":WRONG_SSL_VERSION:",
		},
		{
			protocol:      dtls,
			name:          "DisableEverything-DTLS",
			flags:         []string{"-no-tls12", "-no-tls1"},
			shouldFail:    true,
			expectedError: ":WRONG_SSL_VERSION:",
		},
		{
			name: "NoSharedCipher",
			config: Config{
				CipherSuites: []uint16{},
			},
			shouldFail:    true,
			expectedError: ":HANDSHAKE_FAILURE_ON_CLIENT_HELLO:",
		},
		{
			protocol: dtls,
			testType: serverTest,
			name:     "MTU",
			config: Config{
				Bugs: ProtocolBugs{
					MaxPacketLength: 256,
				},
			},
			flags: []string{"-mtu", "256"},
		},
		{
			protocol: dtls,
			testType: serverTest,
			name:     "MTUExceeded",
			config: Config{
				Bugs: ProtocolBugs{
					MaxPacketLength: 255,
				},
			},
			flags:              []string{"-mtu", "256"},
			shouldFail:         true,
			expectedLocalError: "dtls: exceeded maximum packet length",
		},
		{
			name: "CertMismatchRSA",
			config: Config{
				CipherSuites: []uint16{TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256},
				Certificates: []Certificate{getECDSACertificate()},
				Bugs: ProtocolBugs{
					SendCipherSuite: TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
				},
			},
			shouldFail:    true,
			expectedError: ":WRONG_CERTIFICATE_TYPE:",
		},
		{
			name: "CertMismatchECDSA",
			config: Config{
				CipherSuites: []uint16{TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256},
				Certificates: []Certificate{getRSACertificate()},
				Bugs: ProtocolBugs{
					SendCipherSuite: TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256,
				},
			},
			shouldFail:    true,
			expectedError: ":WRONG_CERTIFICATE_TYPE:",
		},
		{
			name: "EmptyCertificateList",
			config: Config{
				CipherSuites: []uint16{TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256},
				Bugs: ProtocolBugs{
					EmptyCertificateList: true,
				},
			},
			shouldFail:    true,
			expectedError: ":DECODE_ERROR:",
		},
		{
			name:             "TLSFatalBadPackets",
			damageFirstWrite: true,
			shouldFail:       true,
			expectedError:    ":DECRYPTION_FAILED_OR_BAD_RECORD_MAC:",
		},
		{
			protocol:         dtls,
			name:             "DTLSIgnoreBadPackets",
			damageFirstWrite: true,
		},
		{
			protocol:         dtls,
			name:             "DTLSIgnoreBadPackets-Async",
			damageFirstWrite: true,
			flags:            []string{"-async"},
		},
		{
			name: "AppDataBeforeHandshake",
			config: Config{
				Bugs: ProtocolBugs{
					AppDataBeforeHandshake: []byte("TEST MESSAGE"),
				},
			},
			shouldFail:    true,
			expectedError: ":UNEXPECTED_RECORD:",
		},
		{
			name: "AppDataBeforeHandshake-Empty",
			config: Config{
				Bugs: ProtocolBugs{
					AppDataBeforeHandshake: []byte{},
				},
			},
			shouldFail:    true,
			expectedError: ":UNEXPECTED_RECORD:",
		},
		{
			protocol: dtls,
			name:     "AppDataBeforeHandshake-DTLS",
			config: Config{
				Bugs: ProtocolBugs{
					AppDataBeforeHandshake: []byte("TEST MESSAGE"),
				},
			},
			shouldFail:    true,
			expectedError: ":UNEXPECTED_RECORD:",
		},
		{
			protocol: dtls,
			name:     "AppDataBeforeHandshake-DTLS-Empty",
			config: Config{
				Bugs: ProtocolBugs{
					AppDataBeforeHandshake: []byte{},
				},
			},
			shouldFail:    true,
			expectedError: ":UNEXPECTED_RECORD:",
		},
		{
			name: "AppDataAfterChangeCipherSpec",
			config: Config{
				Bugs: ProtocolBugs{
					AppDataAfterChangeCipherSpec: []byte("TEST MESSAGE"),
				},
			},
			shouldFail:    true,
			expectedError: ":DATA_BETWEEN_CCS_AND_FINISHED:",
		},
		{
			name: "AppDataAfterChangeCipherSpec-Empty",
			config: Config{
				Bugs: ProtocolBugs{
					AppDataAfterChangeCipherSpec: []byte{},
				},
			},
			shouldFail:    true,
			expectedError: ":DATA_BETWEEN_CCS_AND_FINISHED:",
		},
		{
			protocol: dtls,
			name:     "AppDataAfterChangeCipherSpec-DTLS",
			config: Config{
				Bugs: ProtocolBugs{
					AppDataAfterChangeCipherSpec: []byte("TEST MESSAGE"),
				},
			},
			// BoringSSL's DTLS implementation will drop the out-of-order
			// application data.
		},
		{
			protocol: dtls,
			name:     "AppDataAfterChangeCipherSpec-DTLS-Empty",
			config: Config{
				Bugs: ProtocolBugs{
					AppDataAfterChangeCipherSpec: []byte{},
				},
			},
			// BoringSSL's DTLS implementation will drop the out-of-order
			// application data.
		},
		{
			name: "AlertAfterChangeCipherSpec",
			config: Config{
				Bugs: ProtocolBugs{
					AlertAfterChangeCipherSpec: alertRecordOverflow,
				},
			},
			shouldFail:    true,
			expectedError: ":TLSV1_ALERT_RECORD_OVERFLOW:",
		},
		{
			protocol: dtls,
			name:     "AlertAfterChangeCipherSpec-DTLS",
			config: Config{
				Bugs: ProtocolBugs{
					AlertAfterChangeCipherSpec: alertRecordOverflow,
				},
			},
			shouldFail:    true,
			expectedError: ":TLSV1_ALERT_RECORD_OVERFLOW:",
		},
		{
			protocol: dtls,
			name:     "ReorderHandshakeFragments-Small-DTLS",
			config: Config{
				Bugs: ProtocolBugs{
					ReorderHandshakeFragments: true,
					// Small enough that every handshake message is
					// fragmented.
					MaxHandshakeRecordLength: 2,
				},
			},
		},
		{
			protocol: dtls,
			name:     "ReorderHandshakeFragments-Large-DTLS",
			config: Config{
				Bugs: ProtocolBugs{
					ReorderHandshakeFragments: true,
					// Large enough that no handshake message is
					// fragmented.
					MaxHandshakeRecordLength: 2048,
				},
			},
		},
		{
			protocol: dtls,
			name:     "MixCompleteMessageWithFragments-DTLS",
			config: Config{
				Bugs: ProtocolBugs{
					ReorderHandshakeFragments:       true,
					MixCompleteMessageWithFragments: true,
					MaxHandshakeRecordLength:        2,
				},
			},
		},
		{
			name: "SendInvalidRecordType",
			config: Config{
				Bugs: ProtocolBugs{
					SendInvalidRecordType: true,
				},
			},
			shouldFail:    true,
			expectedError: ":UNEXPECTED_RECORD:",
		},
		{
			protocol: dtls,
			name:     "SendInvalidRecordType-DTLS",
			config: Config{
				Bugs: ProtocolBugs{
					SendInvalidRecordType: true,
				},
			},
			shouldFail:    true,
			expectedError: ":UNEXPECTED_RECORD:",
		},
		{
			name: "FalseStart-SkipServerSecondLeg",
			config: Config{
				CipherSuites: []uint16{TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256},
				NextProtos:   []string{"foo"},
				Bugs: ProtocolBugs{
					SkipNewSessionTicket: true,
					SkipChangeCipherSpec: true,
					SkipFinished:         true,
					ExpectFalseStart:     true,
				},
			},
			flags: []string{
				"-false-start",
				"-handshake-never-done",
				"-advertise-alpn", "\x03foo",
			},
			shimWritesFirst: true,
			shouldFail:      true,
			expectedError:   ":UNEXPECTED_RECORD:",
		},
		{
			name: "FalseStart-SkipServerSecondLeg-Implicit",
			config: Config{
				CipherSuites: []uint16{TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256},
				NextProtos:   []string{"foo"},
				Bugs: ProtocolBugs{
					SkipNewSessionTicket: true,
					SkipChangeCipherSpec: true,
					SkipFinished:         true,
				},
			},
			flags: []string{
				"-implicit-handshake",
				"-false-start",
				"-handshake-never-done",
				"-advertise-alpn", "\x03foo",
			},
			shouldFail:    true,
			expectedError: ":UNEXPECTED_RECORD:",
		},
		{
			testType:           serverTest,
			name:               "FailEarlyCallback",
			flags:              []string{"-fail-early-callback"},
			shouldFail:         true,
			expectedError:      ":CONNECTION_REJECTED:",
			expectedLocalError: "remote error: access denied",
		},
		{
			name: "WrongMessageType",
			config: Config{
				Bugs: ProtocolBugs{
					WrongCertificateMessageType: true,
				},
			},
			shouldFail:         true,
			expectedError:      ":UNEXPECTED_MESSAGE:",
			expectedLocalError: "remote error: unexpected message",
		},
		{
			protocol: dtls,
			name:     "WrongMessageType-DTLS",
			config: Config{
				Bugs: ProtocolBugs{
					WrongCertificateMessageType: true,
				},
			},
			shouldFail:         true,
			expectedError:      ":UNEXPECTED_MESSAGE:",
			expectedLocalError: "remote error: unexpected message",
		},
		{
			protocol: dtls,
			name:     "FragmentMessageTypeMismatch-DTLS",
			config: Config{
				Bugs: ProtocolBugs{
					MaxHandshakeRecordLength:    2,
					FragmentMessageTypeMismatch: true,
				},
			},
			shouldFail:    true,
			expectedError: ":FRAGMENT_MISMATCH:",
		},
		{
			protocol: dtls,
			name:     "FragmentMessageLengthMismatch-DTLS",
			config: Config{
				Bugs: ProtocolBugs{
					MaxHandshakeRecordLength:      2,
					FragmentMessageLengthMismatch: true,
				},
			},
			shouldFail:    true,
			expectedError: ":FRAGMENT_MISMATCH:",
		},
		{
			protocol: dtls,
			name:     "SplitFragments-Header-DTLS",
			config: Config{
				Bugs: ProtocolBugs{
					SplitFragments: 2,
				},
			},
			shouldFail:    true,
			expectedError: ":UNEXPECTED_MESSAGE:",
		},
		{
			protocol: dtls,
			name:     "SplitFragments-Boundary-DTLS",
			config: Config{
				Bugs: ProtocolBugs{
					SplitFragments: dtlsRecordHeaderLen,
				},
			},
			shouldFail:    true,
			expectedError: ":EXCESSIVE_MESSAGE_SIZE:",
		},
		{
			protocol: dtls,
			name:     "SplitFragments-Body-DTLS",
			config: Config{
				Bugs: ProtocolBugs{
					SplitFragments: dtlsRecordHeaderLen + 1,
				},
			},
			shouldFail:    true,
			expectedError: ":EXCESSIVE_MESSAGE_SIZE:",
		},
		{
			protocol: dtls,
			name:     "SendEmptyFragments-DTLS",
			config: Config{
				Bugs: ProtocolBugs{
					SendEmptyFragments: true,
				},
			},
		},
		{
			name: "UnsupportedCipherSuite",
			config: Config{
				CipherSuites: []uint16{TLS_RSA_WITH_RC4_128_SHA},
				Bugs: ProtocolBugs{
					IgnorePeerCipherPreferences: true,
				},
			},
			flags:         []string{"-cipher", "DEFAULT:!RC4"},
			shouldFail:    true,
			expectedError: ":WRONG_CIPHER_RETURNED:",
		},
		{
			name: "UnsupportedCurve",
			config: Config{
				CipherSuites: []uint16{TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256},
				// BoringSSL implements P-224 but doesn't enable it by
				// default.
				CurvePreferences: []CurveID{CurveP224},
				Bugs: ProtocolBugs{
					IgnorePeerCurvePreferences: true,
				},
			},
			shouldFail:    true,
			expectedError: ":WRONG_CURVE:",
		},
		{
			name: "BadFinished",
			config: Config{
				Bugs: ProtocolBugs{
					BadFinished: true,
				},
			},
			shouldFail:    true,
			expectedError: ":DIGEST_CHECK_FAILED:",
		},
		{
			name: "FalseStart-BadFinished",
			config: Config{
				CipherSuites: []uint16{TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256},
				NextProtos:   []string{"foo"},
				Bugs: ProtocolBugs{
					BadFinished:      true,
					ExpectFalseStart: true,
				},
			},
			flags: []string{
				"-false-start",
				"-handshake-never-done",
				"-advertise-alpn", "\x03foo",
			},
			shimWritesFirst: true,
			shouldFail:      true,
			expectedError:   ":DIGEST_CHECK_FAILED:",
		},
		{
			name: "NoFalseStart-NoALPN",
			config: Config{
				CipherSuites: []uint16{TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256},
				Bugs: ProtocolBugs{
					ExpectFalseStart:          true,
					AlertBeforeFalseStartTest: alertAccessDenied,
				},
			},
			flags: []string{
				"-false-start",
			},
			shimWritesFirst:    true,
			shouldFail:         true,
			expectedError:      ":TLSV1_ALERT_ACCESS_DENIED:",
			expectedLocalError: "tls: peer did not false start: EOF",
		},
		{
			name: "NoFalseStart-NoAEAD",
			config: Config{
				CipherSuites: []uint16{TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA},
				NextProtos:   []string{"foo"},
				Bugs: ProtocolBugs{
					ExpectFalseStart:          true,
					AlertBeforeFalseStartTest: alertAccessDenied,
				},
			},
			flags: []string{
				"-false-start",
				"-advertise-alpn", "\x03foo",
			},
			shimWritesFirst:    true,
			shouldFail:         true,
			expectedError:      ":TLSV1_ALERT_ACCESS_DENIED:",
			expectedLocalError: "tls: peer did not false start: EOF",
		},
		{
			name: "NoFalseStart-RSA",
			config: Config{
				CipherSuites: []uint16{TLS_RSA_WITH_AES_128_GCM_SHA256},
				NextProtos:   []string{"foo"},
				Bugs: ProtocolBugs{
					ExpectFalseStart:          true,
					AlertBeforeFalseStartTest: alertAccessDenied,
				},
			},
			flags: []string{
				"-false-start",
				"-advertise-alpn", "\x03foo",
			},
			shimWritesFirst:    true,
			shouldFail:         true,
			expectedError:      ":TLSV1_ALERT_ACCESS_DENIED:",
			expectedLocalError: "tls: peer did not false start: EOF",
		},
		{
			name: "NoFalseStart-DHE_RSA",
			config: Config{
				CipherSuites: []uint16{TLS_DHE_RSA_WITH_AES_128_GCM_SHA256},
				NextProtos:   []string{"foo"},
				Bugs: ProtocolBugs{
					ExpectFalseStart:          true,
					AlertBeforeFalseStartTest: alertAccessDenied,
				},
			},
			flags: []string{
				"-false-start",
				"-advertise-alpn", "\x03foo",
			},
			shimWritesFirst:    true,
			shouldFail:         true,
			expectedError:      ":TLSV1_ALERT_ACCESS_DENIED:",
			expectedLocalError: "tls: peer did not false start: EOF",
		},
		{
			testType: serverTest,
			name:     "NoSupportedCurves",
			config: Config{
				CipherSuites: []uint16{TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256},
				Bugs: ProtocolBugs{
					NoSupportedCurves: true,
				},
			},
		},
		{
			testType: serverTest,
			name:     "NoCommonCurves",
			config: Config{
				CipherSuites: []uint16{
					TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
					TLS_DHE_RSA_WITH_AES_128_GCM_SHA256,
				},
				CurvePreferences: []CurveID{CurveP224},
			},
			expectedCipher: TLS_DHE_RSA_WITH_AES_128_GCM_SHA256,
		},
		{
			protocol: dtls,
			name:     "SendSplitAlert-Sync",
			config: Config{
				Bugs: ProtocolBugs{
					SendSplitAlert: true,
				},
			},
		},
		{
			protocol: dtls,
			name:     "SendSplitAlert-Async",
			config: Config{
				Bugs: ProtocolBugs{
					SendSplitAlert: true,
				},
			},
			flags: []string{"-async"},
		},
		{
			protocol: dtls,
			name:     "PackDTLSHandshake",
			config: Config{
				Bugs: ProtocolBugs{
					MaxHandshakeRecordLength: 2,
					PackHandshakeFragments:   20,
					PackHandshakeRecords:     200,
				},
			},
		},
		{
			testType: serverTest,
			protocol: dtls,
			name:     "NoRC4-DTLS",
			config: Config{
				CipherSuites: []uint16{TLS_ECDHE_RSA_WITH_RC4_128_SHA},
				Bugs: ProtocolBugs{
					EnableAllCiphersInDTLS: true,
				},
			},
			shouldFail:    true,
			expectedError: ":NO_SHARED_CIPHER:",
		},
		{
			name:             "SendEmptyRecords-Pass",
			sendEmptyRecords: 32,
		},
		{
			name:             "SendEmptyRecords",
			sendEmptyRecords: 33,
			shouldFail:       true,
			expectedError:    ":TOO_MANY_EMPTY_FRAGMENTS:",
		},
		{
			name:             "SendEmptyRecords-Async",
			sendEmptyRecords: 33,
			flags:            []string{"-async"},
			shouldFail:       true,
			expectedError:    ":TOO_MANY_EMPTY_FRAGMENTS:",
		},
		{
			name:              "SendWarningAlerts-Pass",
			sendWarningAlerts: 4,
		},
		{
			protocol:          dtls,
			name:              "SendWarningAlerts-DTLS-Pass",
			sendWarningAlerts: 4,
		},
		{
			name:              "SendWarningAlerts",
			sendWarningAlerts: 5,
			shouldFail:        true,
			expectedError:     ":TOO_MANY_WARNING_ALERTS:",
		},
		{
			name:              "SendWarningAlerts-Async",
			sendWarningAlerts: 5,
			flags:             []string{"-async"},
			shouldFail:        true,
			expectedError:     ":TOO_MANY_WARNING_ALERTS:",
		},
		{
			name: "EmptySessionID",
			config: Config{
				SessionTicketsDisabled: true,
			},
			noSessionCache: true,
			flags:          []string{"-expect-no-session"},
		},
		{
			name: "Unclean-Shutdown",
			config: Config{
				Bugs: ProtocolBugs{
					NoCloseNotify:     true,
					ExpectCloseNotify: true,
				},
			},
			shimShutsDown: true,
			flags:         []string{"-check-close-notify"},
			shouldFail:    true,
			expectedError: "Unexpected SSL_shutdown result: -1 != 1",
		},
		{
			name: "Unclean-Shutdown-Ignored",
			config: Config{
				Bugs: ProtocolBugs{
					NoCloseNotify: true,
				},
			},
			shimShutsDown: true,
		},
		{
			name: "LargePlaintext",
			config: Config{
				Bugs: ProtocolBugs{
					SendLargeRecords: true,
				},
			},
			messageLen:    maxPlaintext + 1,
			shouldFail:    true,
			expectedError: ":DATA_LENGTH_TOO_LONG:",
		},
		{
			protocol: dtls,
			name:     "LargePlaintext-DTLS",
			config: Config{
				Bugs: ProtocolBugs{
					SendLargeRecords: true,
				},
			},
			messageLen:    maxPlaintext + 1,
			shouldFail:    true,
			expectedError: ":DATA_LENGTH_TOO_LONG:",
		},
		{
			name: "LargeCiphertext",
			config: Config{
				Bugs: ProtocolBugs{
					SendLargeRecords: true,
				},
			},
			messageLen:    maxPlaintext * 2,
			shouldFail:    true,
			expectedError: ":ENCRYPTED_LENGTH_TOO_LONG:",
		},
		{
			protocol: dtls,
			name:     "LargeCiphertext-DTLS",
			config: Config{
				Bugs: ProtocolBugs{
					SendLargeRecords: true,
				},
			},
			messageLen: maxPlaintext * 2,
			// Unlike the other four cases, DTLS drops records which
			// are invalid before authentication, so the connection
			// does not fail.
			expectMessageDropped: true,
		},
	}
	testCases = append(testCases, basicTests...)
}

func addCipherSuiteTests() {
	for _, suite := range testCipherSuites {
		const psk = "12345"
		const pskIdentity = "luggage combo"

		var cert Certificate
		var certFile string
		var keyFile string
		if hasComponent(suite.name, "ECDSA") {
			cert = getECDSACertificate()
			certFile = ecdsaCertificateFile
			keyFile = ecdsaKeyFile
		} else {
			cert = getRSACertificate()
			certFile = rsaCertificateFile
			keyFile = rsaKeyFile
		}

		var flags []string
		if hasComponent(suite.name, "PSK") {
			flags = append(flags,
				"-psk", psk,
				"-psk-identity", pskIdentity)
		}
		if hasComponent(suite.name, "NULL") {
			// NULL ciphers must be explicitly enabled.
			flags = append(flags, "-cipher", "DEFAULT:NULL-SHA")
		}

		for _, ver := range tlsVersions {
			if ver.version < VersionTLS12 && isTLS12Only(suite.name) {
				continue
			}

			testCases = append(testCases, testCase{
				testType: clientTest,
				name:     ver.name + "-" + suite.name + "-client",
				config: Config{
					MinVersion:           ver.version,
					MaxVersion:           ver.version,
					CipherSuites:         []uint16{suite.id},
					Certificates:         []Certificate{cert},
					PreSharedKey:         []byte(psk),
					PreSharedKeyIdentity: pskIdentity,
				},
				flags:         flags,
				resumeSession: true,
			})

			testCases = append(testCases, testCase{
				testType: serverTest,
				name:     ver.name + "-" + suite.name + "-server",
				config: Config{
					MinVersion:           ver.version,
					MaxVersion:           ver.version,
					CipherSuites:         []uint16{suite.id},
					Certificates:         []Certificate{cert},
					PreSharedKey:         []byte(psk),
					PreSharedKeyIdentity: pskIdentity,
				},
				certFile:      certFile,
				keyFile:       keyFile,
				flags:         flags,
				resumeSession: true,
			})

			if ver.hasDTLS && isDTLSCipher(suite.name) {
				testCases = append(testCases, testCase{
					testType: clientTest,
					protocol: dtls,
					name:     "D" + ver.name + "-" + suite.name + "-client",
					config: Config{
						MinVersion:           ver.version,
						MaxVersion:           ver.version,
						CipherSuites:         []uint16{suite.id},
						Certificates:         []Certificate{cert},
						PreSharedKey:         []byte(psk),
						PreSharedKeyIdentity: pskIdentity,
					},
					flags:         flags,
					resumeSession: true,
				})
				testCases = append(testCases, testCase{
					testType: serverTest,
					protocol: dtls,
					name:     "D" + ver.name + "-" + suite.name + "-server",
					config: Config{
						MinVersion:           ver.version,
						MaxVersion:           ver.version,
						CipherSuites:         []uint16{suite.id},
						Certificates:         []Certificate{cert},
						PreSharedKey:         []byte(psk),
						PreSharedKeyIdentity: pskIdentity,
					},
					certFile:      certFile,
					keyFile:       keyFile,
					flags:         flags,
					resumeSession: true,
				})
			}
		}

		// Ensure both TLS and DTLS accept their maximum record sizes.
		testCases = append(testCases, testCase{
			name: suite.name + "-LargeRecord",
			config: Config{
				CipherSuites:         []uint16{suite.id},
				Certificates:         []Certificate{cert},
				PreSharedKey:         []byte(psk),
				PreSharedKeyIdentity: pskIdentity,
			},
			flags:      flags,
			messageLen: maxPlaintext,
		})
		testCases = append(testCases, testCase{
			name: suite.name + "-LargeRecord-Extra",
			config: Config{
				CipherSuites:         []uint16{suite.id},
				Certificates:         []Certificate{cert},
				PreSharedKey:         []byte(psk),
				PreSharedKeyIdentity: pskIdentity,
				Bugs: ProtocolBugs{
					SendLargeRecords: true,
				},
			},
			flags:      append(flags, "-microsoft-big-sslv3-buffer"),
			messageLen: maxPlaintext + 16384,
		})
		if isDTLSCipher(suite.name) {
			testCases = append(testCases, testCase{
				protocol: dtls,
				name:     suite.name + "-LargeRecord-DTLS",
				config: Config{
					CipherSuites:         []uint16{suite.id},
					Certificates:         []Certificate{cert},
					PreSharedKey:         []byte(psk),
					PreSharedKeyIdentity: pskIdentity,
				},
				flags:      flags,
				messageLen: maxPlaintext,
			})
		}
	}

	testCases = append(testCases, testCase{
		name: "WeakDH",
		config: Config{
			CipherSuites: []uint16{TLS_DHE_RSA_WITH_AES_128_GCM_SHA256},
			Bugs: ProtocolBugs{
				// This is a 1023-bit prime number, generated
				// with:
				// openssl gendh 1023 | openssl asn1parse -i
				DHGroupPrime: bigFromHex("518E9B7930CE61C6E445C8360584E5FC78D9137C0FFDC880B495D5338ADF7689951A6821C17A76B3ACB8E0156AEA607B7EC406EBEDBB84D8376EB8FE8F8BA1433488BEE0C3EDDFD3A32DBB9481980A7AF6C96BFCF490A094CFFB2B8192C1BB5510B77B658436E27C2D4D023FE3718222AB0CA1273995B51F6D625A4944D0DD4B"),
			},
		},
		shouldFail:    true,
		expectedError: "BAD_DH_P_LENGTH",
	})

	// versionSpecificCiphersTest specifies a test for the TLS 1.0 and TLS
	// 1.1 specific cipher suite settings. A server is setup with the given
	// cipher lists and then a connection is made for each member of
	// expectations. The cipher suite that the server selects must match
	// the specified one.
	var versionSpecificCiphersTest = []struct {
		ciphersDefault, ciphersTLS10, ciphersTLS11 string
		// expectations is a map from TLS version to cipher suite id.
		expectations map[uint16]uint16
	}{
		{
			// Test that the null case (where no version-specific ciphers are set)
			// works as expected.
			"RC4-SHA:AES128-SHA", // default ciphers
			"",                   // no ciphers specifically for TLS ≥ 1.0
			"",                   // no ciphers specifically for TLS ≥ 1.1
			map[uint16]uint16{
				VersionSSL30: TLS_RSA_WITH_RC4_128_SHA,
				VersionTLS10: TLS_RSA_WITH_RC4_128_SHA,
				VersionTLS11: TLS_RSA_WITH_RC4_128_SHA,
				VersionTLS12: TLS_RSA_WITH_RC4_128_SHA,
			},
		},
		{
			// With ciphers_tls10 set, TLS 1.0, 1.1 and 1.2 should get a different
			// cipher.
			"RC4-SHA:AES128-SHA", // default
			"AES128-SHA",         // these ciphers for TLS ≥ 1.0
			"",                   // no ciphers specifically for TLS ≥ 1.1
			map[uint16]uint16{
				VersionSSL30: TLS_RSA_WITH_RC4_128_SHA,
				VersionTLS10: TLS_RSA_WITH_AES_128_CBC_SHA,
				VersionTLS11: TLS_RSA_WITH_AES_128_CBC_SHA,
				VersionTLS12: TLS_RSA_WITH_AES_128_CBC_SHA,
			},
		},
		{
			// With ciphers_tls11 set, TLS 1.1 and 1.2 should get a different
			// cipher.
			"RC4-SHA:AES128-SHA", // default
			"",                   // no ciphers specifically for TLS ≥ 1.0
			"AES128-SHA",         // these ciphers for TLS ≥ 1.1
			map[uint16]uint16{
				VersionSSL30: TLS_RSA_WITH_RC4_128_SHA,
				VersionTLS10: TLS_RSA_WITH_RC4_128_SHA,
				VersionTLS11: TLS_RSA_WITH_AES_128_CBC_SHA,
				VersionTLS12: TLS_RSA_WITH_AES_128_CBC_SHA,
			},
		},
		{
			// With both ciphers_tls10 and ciphers_tls11 set, ciphers_tls11 should
			// mask ciphers_tls10 for TLS 1.1 and 1.2.
			"RC4-SHA:AES128-SHA", // default
			"AES128-SHA",         // these ciphers for TLS ≥ 1.0
			"AES256-SHA",         // these ciphers for TLS ≥ 1.1
			map[uint16]uint16{
				VersionSSL30: TLS_RSA_WITH_RC4_128_SHA,
				VersionTLS10: TLS_RSA_WITH_AES_128_CBC_SHA,
				VersionTLS11: TLS_RSA_WITH_AES_256_CBC_SHA,
				VersionTLS12: TLS_RSA_WITH_AES_256_CBC_SHA,
			},
		},
	}

	for i, test := range versionSpecificCiphersTest {
		for version, expectedCipherSuite := range test.expectations {
			flags := []string{"-cipher", test.ciphersDefault}
			if len(test.ciphersTLS10) > 0 {
				flags = append(flags, "-cipher-tls10", test.ciphersTLS10)
			}
			if len(test.ciphersTLS11) > 0 {
				flags = append(flags, "-cipher-tls11", test.ciphersTLS11)
			}

			testCases = append(testCases, testCase{
				testType: serverTest,
				name:     fmt.Sprintf("VersionSpecificCiphersTest-%d-%x", i, version),
				config: Config{
					MaxVersion:   version,
					MinVersion:   version,
					CipherSuites: []uint16{TLS_RSA_WITH_RC4_128_SHA, TLS_RSA_WITH_AES_128_CBC_SHA, TLS_RSA_WITH_AES_256_CBC_SHA},
				},
				flags:          flags,
				expectedCipher: expectedCipherSuite,
			})
		}
	}
}

func addBadECDSASignatureTests() {
	for badR := BadValue(1); badR < NumBadValues; badR++ {
		for badS := BadValue(1); badS < NumBadValues; badS++ {
			testCases = append(testCases, testCase{
				name: fmt.Sprintf("BadECDSA-%d-%d", badR, badS),
				config: Config{
					CipherSuites: []uint16{TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256},
					Certificates: []Certificate{getECDSACertificate()},
					Bugs: ProtocolBugs{
						BadECDSAR: badR,
						BadECDSAS: badS,
					},
				},
				shouldFail:    true,
				expectedError: "SIGNATURE",
			})
		}
	}
}

func addCBCPaddingTests() {
	testCases = append(testCases, testCase{
		name: "MaxCBCPadding",
		config: Config{
			CipherSuites: []uint16{TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA},
			Bugs: ProtocolBugs{
				MaxPadding: true,
			},
		},
		messageLen: 12, // 20 bytes of SHA-1 + 12 == 0 % block size
	})
	testCases = append(testCases, testCase{
		name: "BadCBCPadding",
		config: Config{
			CipherSuites: []uint16{TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA},
			Bugs: ProtocolBugs{
				PaddingFirstByteBad: true,
			},
		},
		shouldFail:    true,
		expectedError: "DECRYPTION_FAILED_OR_BAD_RECORD_MAC",
	})
	// OpenSSL previously had an issue where the first byte of padding in
	// 255 bytes of padding wasn't checked.
	testCases = append(testCases, testCase{
		name: "BadCBCPadding255",
		config: Config{
			CipherSuites: []uint16{TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA},
			Bugs: ProtocolBugs{
				MaxPadding:               true,
				PaddingFirstByteBadIf255: true,
			},
		},
		messageLen:    12, // 20 bytes of SHA-1 + 12 == 0 % block size
		shouldFail:    true,
		expectedError: "DECRYPTION_FAILED_OR_BAD_RECORD_MAC",
	})
}

func addCBCSplittingTests() {
	testCases = append(testCases, testCase{
		name: "CBCRecordSplitting",
		config: Config{
			MaxVersion:   VersionTLS10,
			MinVersion:   VersionTLS10,
			CipherSuites: []uint16{TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA},
		},
		messageLen:    -1, // read until EOF
		resumeSession: true,
		flags: []string{
			"-async",
			"-write-different-record-sizes",
			"-cbc-record-splitting",
		},
	})
	testCases = append(testCases, testCase{
		name: "CBCRecordSplittingPartialWrite",
		config: Config{
			MaxVersion:   VersionTLS10,
			MinVersion:   VersionTLS10,
			CipherSuites: []uint16{TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA},
		},
		messageLen: -1, // read until EOF
		flags: []string{
			"-async",
			"-write-different-record-sizes",
			"-cbc-record-splitting",
			"-partial-write",
		},
	})
}

func addClientAuthTests() {
	// Add a dummy cert pool to stress certificate authority parsing.
	// TODO(davidben): Add tests that those values parse out correctly.
	certPool := x509.NewCertPool()
	cert, err := x509.ParseCertificate(rsaCertificate.Certificate[0])
	if err != nil {
		panic(err)
	}
	certPool.AddCert(cert)

	for _, ver := range tlsVersions {
		testCases = append(testCases, testCase{
			testType: clientTest,
			name:     ver.name + "-Client-ClientAuth-RSA",
			config: Config{
				MinVersion: ver.version,
				MaxVersion: ver.version,
				ClientAuth: RequireAnyClientCert,
				ClientCAs:  certPool,
			},
			flags: []string{
				"-cert-file", path.Join(*resourceDir, rsaCertificateFile),
				"-key-file", path.Join(*resourceDir, rsaKeyFile),
			},
		})
		testCases = append(testCases, testCase{
			testType: serverTest,
			name:     ver.name + "-Server-ClientAuth-RSA",
			config: Config{
				MinVersion:   ver.version,
				MaxVersion:   ver.version,
				Certificates: []Certificate{rsaCertificate},
			},
			flags: []string{"-require-any-client-certificate"},
		})
		if ver.version != VersionSSL30 {
			testCases = append(testCases, testCase{
				testType: serverTest,
				name:     ver.name + "-Server-ClientAuth-ECDSA",
				config: Config{
					MinVersion:   ver.version,
					MaxVersion:   ver.version,
					Certificates: []Certificate{ecdsaCertificate},
				},
				flags: []string{"-require-any-client-certificate"},
			})
			testCases = append(testCases, testCase{
				testType: clientTest,
				name:     ver.name + "-Client-ClientAuth-ECDSA",
				config: Config{
					MinVersion: ver.version,
					MaxVersion: ver.version,
					ClientAuth: RequireAnyClientCert,
					ClientCAs:  certPool,
				},
				flags: []string{
					"-cert-file", path.Join(*resourceDir, ecdsaCertificateFile),
					"-key-file", path.Join(*resourceDir, ecdsaKeyFile),
				},
			})
		}
	}
}

func addExtendedMasterSecretTests() {
	const expectEMSFlag = "-expect-extended-master-secret"

	for _, with := range []bool{false, true} {
		prefix := "No"
		var flags []string
		if with {
			prefix = ""
			flags = []string{expectEMSFlag}
		}

		for _, isClient := range []bool{false, true} {
			suffix := "-Server"
			testType := serverTest
			if isClient {
				suffix = "-Client"
				testType = clientTest
			}

			for _, ver := range tlsVersions {
				test := testCase{
					testType: testType,
					name:     prefix + "ExtendedMasterSecret-" + ver.name + suffix,
					config: Config{
						MinVersion: ver.version,
						MaxVersion: ver.version,
						Bugs: ProtocolBugs{
							NoExtendedMasterSecret:      !with,
							RequireExtendedMasterSecret: with,
						},
					},
					flags:      flags,
					shouldFail: ver.version == VersionSSL30 && with,
				}
				if test.shouldFail {
					test.expectedLocalError = "extended master secret required but not supported by peer"
				}
				testCases = append(testCases, test)
			}
		}
	}

	for _, isClient := range []bool{false, true} {
		for _, supportedInFirstConnection := range []bool{false, true} {
			for _, supportedInResumeConnection := range []bool{false, true} {
				boolToWord := func(b bool) string {
					if b {
						return "Yes"
					}
					return "No"
				}
				suffix := boolToWord(supportedInFirstConnection) + "To" + boolToWord(supportedInResumeConnection) + "-"
				if isClient {
					suffix += "Client"
				} else {
					suffix += "Server"
				}

				supportedConfig := Config{
					Bugs: ProtocolBugs{
						RequireExtendedMasterSecret: true,
					},
				}

				noSupportConfig := Config{
					Bugs: ProtocolBugs{
						NoExtendedMasterSecret: true,
					},
				}

				test := testCase{
					name:          "ExtendedMasterSecret-" + suffix,
					resumeSession: true,
				}

				if !isClient {
					test.testType = serverTest
				}

				if supportedInFirstConnection {
					test.config = supportedConfig
				} else {
					test.config = noSupportConfig
				}

				if supportedInResumeConnection {
					test.resumeConfig = &supportedConfig
				} else {
					test.resumeConfig = &noSupportConfig
				}

				switch suffix {
				case "YesToYes-Client", "YesToYes-Server":
					// When a session is resumed, it should
					// still be aware that its master
					// secret was generated via EMS and
					// thus it's safe to use tls-unique.
					test.flags = []string{expectEMSFlag}
				case "NoToYes-Server":
					// If an original connection did not
					// contain EMS, but a resumption
					// handshake does, then a server should
					// not resume the session.
					test.expectResumeRejected = true
				case "YesToNo-Server":
					// Resuming an EMS session without the
					// EMS extension should cause the
					// server to abort the connection.
					test.shouldFail = true
					test.expectedError = ":RESUMED_EMS_SESSION_WITHOUT_EMS_EXTENSION:"
				case "NoToYes-Client":
					// A client should abort a connection
					// where the server resumed a non-EMS
					// session but echoed the EMS
					// extension.
					test.shouldFail = true
					test.expectedError = ":RESUMED_NON_EMS_SESSION_WITH_EMS_EXTENSION:"
				case "YesToNo-Client":
					// A client should abort a connection
					// where the server didn't echo EMS
					// when the session used it.
					test.shouldFail = true
					test.expectedError = ":RESUMED_EMS_SESSION_WITHOUT_EMS_EXTENSION:"
				}

				testCases = append(testCases, test)
			}
		}
	}
}

// Adds tests that try to cover the range of the handshake state machine, under
// various conditions. Some of these are redundant with other tests, but they
// only cover the synchronous case.
func addStateMachineCoverageTests(async, splitHandshake bool, protocol protocol) {
	var tests []testCase

	// Basic handshake, with resumption. Client and server,
	// session ID and session ticket.
	tests = append(tests, testCase{
		name:          "Basic-Client",
		resumeSession: true,
	})
	tests = append(tests, testCase{
		name: "Basic-Client-RenewTicket",
		config: Config{
			Bugs: ProtocolBugs{
				RenewTicketOnResume: true,
			},
		},
		flags:         []string{"-expect-ticket-renewal"},
		resumeSession: true,
	})
	tests = append(tests, testCase{
		name: "Basic-Client-NoTicket",
		config: Config{
			SessionTicketsDisabled: true,
		},
		resumeSession: true,
	})
	tests = append(tests, testCase{
		name:          "Basic-Client-Implicit",
		flags:         []string{"-implicit-handshake"},
		resumeSession: true,
	})
	tests = append(tests, testCase{
		testType:      serverTest,
		name:          "Basic-Server",
		resumeSession: true,
	})
	tests = append(tests, testCase{
		testType: serverTest,
		name:     "Basic-Server-NoTickets",
		config: Config{
			SessionTicketsDisabled: true,
		},
		resumeSession: true,
	})
	tests = append(tests, testCase{
		testType:      serverTest,
		name:          "Basic-Server-Implicit",
		flags:         []string{"-implicit-handshake"},
		resumeSession: true,
	})
	tests = append(tests, testCase{
		testType:      serverTest,
		name:          "Basic-Server-EarlyCallback",
		flags:         []string{"-use-early-callback"},
		resumeSession: true,
	})

	// TLS client auth.
	tests = append(tests, testCase{
		testType: clientTest,
		name:     "ClientAuth-Client",
		config: Config{
			ClientAuth: RequireAnyClientCert,
		},
		flags: []string{
			"-cert-file", path.Join(*resourceDir, rsaCertificateFile),
			"-key-file", path.Join(*resourceDir, rsaKeyFile),
		},
	})
	if async {
		tests = append(tests, testCase{
			testType: clientTest,
			name:     "ClientAuth-Client-AsyncKey",
			config: Config{
				ClientAuth: RequireAnyClientCert,
			},
			flags: []string{
				"-cert-file", path.Join(*resourceDir, rsaCertificateFile),
				"-key-file", path.Join(*resourceDir, rsaKeyFile),
				"-use-async-private-key",
			},
		})
		tests = append(tests, testCase{
			testType: serverTest,
			name:     "Basic-Server-RSAAsyncKey",
			flags: []string{
				"-cert-file", path.Join(*resourceDir, rsaCertificateFile),
				"-key-file", path.Join(*resourceDir, rsaKeyFile),
				"-use-async-private-key",
			},
		})
		tests = append(tests, testCase{
			testType: serverTest,
			name:     "Basic-Server-ECDSAAsyncKey",
			flags: []string{
				"-cert-file", path.Join(*resourceDir, ecdsaCertificateFile),
				"-key-file", path.Join(*resourceDir, ecdsaKeyFile),
				"-use-async-private-key",
			},
		})
	}
	tests = append(tests, testCase{
		testType: serverTest,
		name:     "ClientAuth-Server",
		config: Config{
			Certificates: []Certificate{rsaCertificate},
		},
		flags: []string{"-require-any-client-certificate"},
	})

	// No session ticket support; server doesn't send NewSessionTicket.
	tests = append(tests, testCase{
		name: "SessionTicketsDisabled-Client",
		config: Config{
			SessionTicketsDisabled: true,
		},
	})
	tests = append(tests, testCase{
		testType: serverTest,
		name:     "SessionTicketsDisabled-Server",
		config: Config{
			SessionTicketsDisabled: true,
		},
	})

	// Skip ServerKeyExchange in PSK key exchange if there's no
	// identity hint.
	tests = append(tests, testCase{
		name: "EmptyPSKHint-Client",
		config: Config{
			CipherSuites: []uint16{TLS_PSK_WITH_AES_128_CBC_SHA},
			PreSharedKey: []byte("secret"),
		},
		flags: []string{"-psk", "secret"},
	})
	tests = append(tests, testCase{
		testType: serverTest,
		name:     "EmptyPSKHint-Server",
		config: Config{
			CipherSuites: []uint16{TLS_PSK_WITH_AES_128_CBC_SHA},
			PreSharedKey: []byte("secret"),
		},
		flags: []string{"-psk", "secret"},
	})

	tests = append(tests, testCase{
		testType: clientTest,
		name:     "OCSPStapling-Client",
		flags: []string{
			"-enable-ocsp-stapling",
			"-expect-ocsp-response",
			base64.StdEncoding.EncodeToString(testOCSPResponse),
			"-verify-peer",
		},
		resumeSession: true,
	})

	tests = append(tests, testCase{
		testType:             serverTest,
		name:                 "OCSPStapling-Server",
		expectedOCSPResponse: testOCSPResponse,
		flags: []string{
			"-ocsp-response",
			base64.StdEncoding.EncodeToString(testOCSPResponse),
		},
		resumeSession: true,
	})

	tests = append(tests, testCase{
		testType: clientTest,
		name:     "CertificateVerificationSucceed",
		flags: []string{
			"-verify-peer",
		},
	})

	tests = append(tests, testCase{
		testType: clientTest,
		name:     "CertificateVerificationFail",
		flags: []string{
			"-verify-fail",
			"-verify-peer",
		},
		shouldFail:    true,
		expectedError: ":CERTIFICATE_VERIFY_FAILED:",
	})

	tests = append(tests, testCase{
		testType: clientTest,
		name:     "CertificateVerificationSoftFail",
		flags: []string{
			"-verify-fail",
			"-expect-verify-result",
		},
	})

	if protocol == tls {
		tests = append(tests, testCase{
			name:        "Renegotiate-Client",
			renegotiate: true,
		})
		// NPN on client and server; results in post-handshake message.
		tests = append(tests, testCase{
			name: "NPN-Client",
			config: Config{
				NextProtos: []string{"foo"},
			},
			flags:                 []string{"-select-next-proto", "foo"},
			expectedNextProto:     "foo",
			expectedNextProtoType: npn,
		})
		tests = append(tests, testCase{
			testType: serverTest,
			name:     "NPN-Server",
			config: Config{
				NextProtos: []string{"bar"},
			},
			flags: []string{
				"-advertise-npn", "\x03foo\x03bar\x03baz",
				"-expect-next-proto", "bar",
			},
			expectedNextProto:     "bar",
			expectedNextProtoType: npn,
		})

		// TODO(davidben): Add tests for when False Start doesn't trigger.

		// Client does False Start and negotiates NPN.
		tests = append(tests, testCase{
			name: "FalseStart",
			config: Config{
				CipherSuites: []uint16{TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256},
				NextProtos:   []string{"foo"},
				Bugs: ProtocolBugs{
					ExpectFalseStart: true,
				},
			},
			flags: []string{
				"-false-start",
				"-select-next-proto", "foo",
			},
			shimWritesFirst: true,
			resumeSession:   true,
		})

		// Client does False Start and negotiates ALPN.
		tests = append(tests, testCase{
			name: "FalseStart-ALPN",
			config: Config{
				CipherSuites: []uint16{TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256},
				NextProtos:   []string{"foo"},
				Bugs: ProtocolBugs{
					ExpectFalseStart: true,
				},
			},
			flags: []string{
				"-false-start",
				"-advertise-alpn", "\x03foo",
			},
			shimWritesFirst: true,
			resumeSession:   true,
		})

		// Client does False Start but doesn't explicitly call
		// SSL_connect.
		tests = append(tests, testCase{
			name: "FalseStart-Implicit",
			config: Config{
				CipherSuites: []uint16{TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256},
				NextProtos:   []string{"foo"},
			},
			flags: []string{
				"-implicit-handshake",
				"-false-start",
				"-advertise-alpn", "\x03foo",
			},
		})

		// False Start without session tickets.
		tests = append(tests, testCase{
			name: "FalseStart-SessionTicketsDisabled",
			config: Config{
				CipherSuites:           []uint16{TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256},
				NextProtos:             []string{"foo"},
				SessionTicketsDisabled: true,
				Bugs: ProtocolBugs{
					ExpectFalseStart: true,
				},
			},
			flags: []string{
				"-false-start",
				"-select-next-proto", "foo",
			},
			shimWritesFirst: true,
		})

		// Server parses a V2ClientHello.
		tests = append(tests, testCase{
			testType: serverTest,
			name:     "SendV2ClientHello",
			config: Config{
				// Choose a cipher suite that does not involve
				// elliptic curves, so no extensions are
				// involved.
				CipherSuites: []uint16{TLS_RSA_WITH_RC4_128_SHA},
				Bugs: ProtocolBugs{
					SendV2ClientHello: true,
				},
			},
		})

		// Client sends a Channel ID.
		tests = append(tests, testCase{
			name: "ChannelID-Client",
			config: Config{
				RequestChannelID: true,
			},
			flags:           []string{"-send-channel-id", path.Join(*resourceDir, channelIDKeyFile)},
			resumeSession:   true,
			expectChannelID: true,
		})

		// Server accepts a Channel ID.
		tests = append(tests, testCase{
			testType: serverTest,
			name:     "ChannelID-Server",
			config: Config{
				ChannelID: channelIDKey,
			},
			flags: []string{
				"-expect-channel-id",
				base64.StdEncoding.EncodeToString(channelIDBytes),
			},
			resumeSession:   true,
			expectChannelID: true,
		})

		// Bidirectional shutdown with the runner initiating.
		tests = append(tests, testCase{
			name: "Shutdown-Runner",
			config: Config{
				Bugs: ProtocolBugs{
					ExpectCloseNotify: true,
				},
			},
			flags: []string{"-check-close-notify"},
		})

		// Bidirectional shutdown with the shim initiating. The runner,
		// in the meantime, sends garbage before the close_notify which
		// the shim must ignore.
		tests = append(tests, testCase{
			name: "Shutdown-Shim",
			config: Config{
				Bugs: ProtocolBugs{
					ExpectCloseNotify: true,
				},
			},
			shimShutsDown:     true,
			sendEmptyRecords:  1,
			sendWarningAlerts: 1,
			flags:             []string{"-check-close-notify"},
		})
	} else {
		tests = append(tests, testCase{
			name: "SkipHelloVerifyRequest",
			config: Config{
				Bugs: ProtocolBugs{
					SkipHelloVerifyRequest: true,
				},
			},
		})
	}

	var suffix string
	var flags []string
	var maxHandshakeRecordLength int
	if protocol == dtls {
		suffix = "-DTLS"
	}
	if async {
		suffix += "-Async"
		flags = append(flags, "-async")
	} else {
		suffix += "-Sync"
	}
	if splitHandshake {
		suffix += "-SplitHandshakeRecords"
		maxHandshakeRecordLength = 1
	}
	for _, test := range tests {
		test.protocol = protocol
		test.name += suffix
		test.config.Bugs.MaxHandshakeRecordLength = maxHandshakeRecordLength
		test.flags = append(test.flags, flags...)
		testCases = append(testCases, test)
	}
}

func addDDoSCallbackTests() {
	// DDoS callback.

	for _, resume := range []bool{false, true} {
		suffix := "Resume"
		if resume {
			suffix = "No" + suffix
		}

		testCases = append(testCases, testCase{
			testType:      serverTest,
			name:          "Server-DDoS-OK-" + suffix,
			flags:         []string{"-install-ddos-callback"},
			resumeSession: resume,
		})

		failFlag := "-fail-ddos-callback"
		if resume {
			failFlag = "-fail-second-ddos-callback"
		}
		testCases = append(testCases, testCase{
			testType:      serverTest,
			name:          "Server-DDoS-Reject-" + suffix,
			flags:         []string{"-install-ddos-callback", failFlag},
			resumeSession: resume,
			shouldFail:    true,
			expectedError: ":CONNECTION_REJECTED:",
		})
	}
}

func addVersionNegotiationTests() {
	for i, shimVers := range tlsVersions {
		// Assemble flags to disable all newer versions on the shim.
		var flags []string
		for _, vers := range tlsVersions[i+1:] {
			flags = append(flags, vers.flag)
		}

		for _, runnerVers := range tlsVersions {
			protocols := []protocol{tls}
			if runnerVers.hasDTLS && shimVers.hasDTLS {
				protocols = append(protocols, dtls)
			}
			for _, protocol := range protocols {
				expectedVersion := shimVers.version
				if runnerVers.version < shimVers.version {
					expectedVersion = runnerVers.version
				}

				suffix := shimVers.name + "-" + runnerVers.name
				if protocol == dtls {
					suffix += "-DTLS"
				}

				shimVersFlag := strconv.Itoa(int(versionToWire(shimVers.version, protocol == dtls)))

				clientVers := shimVers.version
				if clientVers > VersionTLS10 {
					clientVers = VersionTLS10
				}
				testCases = append(testCases, testCase{
					protocol: protocol,
					testType: clientTest,
					name:     "VersionNegotiation-Client-" + suffix,
					config: Config{
						MaxVersion: runnerVers.version,
						Bugs: ProtocolBugs{
							ExpectInitialRecordVersion: clientVers,
						},
					},
					flags:           flags,
					expectedVersion: expectedVersion,
				})
				testCases = append(testCases, testCase{
					protocol: protocol,
					testType: clientTest,
					name:     "VersionNegotiation-Client2-" + suffix,
					config: Config{
						MaxVersion: runnerVers.version,
						Bugs: ProtocolBugs{
							ExpectInitialRecordVersion: clientVers,
						},
					},
					flags:           []string{"-max-version", shimVersFlag},
					expectedVersion: expectedVersion,
				})

				testCases = append(testCases, testCase{
					protocol: protocol,
					testType: serverTest,
					name:     "VersionNegotiation-Server-" + suffix,
					config: Config{
						MaxVersion: runnerVers.version,
						Bugs: ProtocolBugs{
							ExpectInitialRecordVersion: expectedVersion,
						},
					},
					flags:           flags,
					expectedVersion: expectedVersion,
				})
				testCases = append(testCases, testCase{
					protocol: protocol,
					testType: serverTest,
					name:     "VersionNegotiation-Server2-" + suffix,
					config: Config{
						MaxVersion: runnerVers.version,
						Bugs: ProtocolBugs{
							ExpectInitialRecordVersion: expectedVersion,
						},
					},
					flags:           []string{"-max-version", shimVersFlag},
					expectedVersion: expectedVersion,
				})
			}
		}
	}
}

func addMinimumVersionTests() {
	for i, shimVers := range tlsVersions {
		// Assemble flags to disable all older versions on the shim.
		var flags []string
		for _, vers := range tlsVersions[:i] {
			flags = append(flags, vers.flag)
		}

		for _, runnerVers := range tlsVersions {
			protocols := []protocol{tls}
			if runnerVers.hasDTLS && shimVers.hasDTLS {
				protocols = append(protocols, dtls)
			}
			for _, protocol := range protocols {
				suffix := shimVers.name + "-" + runnerVers.name
				if protocol == dtls {
					suffix += "-DTLS"
				}
				shimVersFlag := strconv.Itoa(int(versionToWire(shimVers.version, protocol == dtls)))

				var expectedVersion uint16
				var shouldFail bool
				var expectedError string
				var expectedLocalError string
				if runnerVers.version >= shimVers.version {
					expectedVersion = runnerVers.version
				} else {
					shouldFail = true
					expectedError = ":UNSUPPORTED_PROTOCOL:"
					if runnerVers.version > VersionSSL30 {
						expectedLocalError = "remote error: protocol version not supported"
					} else {
						expectedLocalError = "remote error: handshake failure"
					}
				}

				testCases = append(testCases, testCase{
					protocol: protocol,
					testType: clientTest,
					name:     "MinimumVersion-Client-" + suffix,
					config: Config{
						MaxVersion: runnerVers.version,
					},
					flags:              flags,
					expectedVersion:    expectedVersion,
					shouldFail:         shouldFail,
					expectedError:      expectedError,
					expectedLocalError: expectedLocalError,
				})
				testCases = append(testCases, testCase{
					protocol: protocol,
					testType: clientTest,
					name:     "MinimumVersion-Client2-" + suffix,
					config: Config{
						MaxVersion: runnerVers.version,
					},
					flags:              []string{"-min-version", shimVersFlag},
					expectedVersion:    expectedVersion,
					shouldFail:         shouldFail,
					expectedError:      expectedError,
					expectedLocalError: expectedLocalError,
				})

				testCases = append(testCases, testCase{
					protocol: protocol,
					testType: serverTest,
					name:     "MinimumVersion-Server-" + suffix,
					config: Config{
						MaxVersion: runnerVers.version,
					},
					flags:              flags,
					expectedVersion:    expectedVersion,
					shouldFail:         shouldFail,
					expectedError:      expectedError,
					expectedLocalError: expectedLocalError,
				})
				testCases = append(testCases, testCase{
					protocol: protocol,
					testType: serverTest,
					name:     "MinimumVersion-Server2-" + suffix,
					config: Config{
						MaxVersion: runnerVers.version,
					},
					flags:              []string{"-min-version", shimVersFlag},
					expectedVersion:    expectedVersion,
					shouldFail:         shouldFail,
					expectedError:      expectedError,
					expectedLocalError: expectedLocalError,
				})
			}
		}
	}
}

func addD5BugTests() {
	testCases = append(testCases, testCase{
		testType: serverTest,
		name:     "D5Bug-NoQuirk-Reject",
		config: Config{
			CipherSuites: []uint16{TLS_RSA_WITH_AES_128_GCM_SHA256},
			Bugs: ProtocolBugs{
				SSL3RSAKeyExchange: true,
			},
		},
		shouldFail:    true,
		expectedError: ":TLS_RSA_ENCRYPTED_VALUE_LENGTH_IS_WRONG:",
	})
	testCases = append(testCases, testCase{
		testType: serverTest,
		name:     "D5Bug-Quirk-Normal",
		config: Config{
			CipherSuites: []uint16{TLS_RSA_WITH_AES_128_GCM_SHA256},
		},
		flags: []string{"-tls-d5-bug"},
	})
	testCases = append(testCases, testCase{
		testType: serverTest,
		name:     "D5Bug-Quirk-Bug",
		config: Config{
			CipherSuites: []uint16{TLS_RSA_WITH_AES_128_GCM_SHA256},
			Bugs: ProtocolBugs{
				SSL3RSAKeyExchange: true,
			},
		},
		flags: []string{"-tls-d5-bug"},
	})
}

func addExtensionTests() {
	testCases = append(testCases, testCase{
		testType: clientTest,
		name:     "DuplicateExtensionClient",
		config: Config{
			Bugs: ProtocolBugs{
				DuplicateExtension: true,
			},
		},
		shouldFail:         true,
		expectedLocalError: "remote error: error decoding message",
	})
	testCases = append(testCases, testCase{
		testType: serverTest,
		name:     "DuplicateExtensionServer",
		config: Config{
			Bugs: ProtocolBugs{
				DuplicateExtension: true,
			},
		},
		shouldFail:         true,
		expectedLocalError: "remote error: error decoding message",
	})
	testCases = append(testCases, testCase{
		testType: clientTest,
		name:     "ServerNameExtensionClient",
		config: Config{
			Bugs: ProtocolBugs{
				ExpectServerName: "example.com",
			},
		},
		flags: []string{"-host-name", "example.com"},
	})
	testCases = append(testCases, testCase{
		testType: clientTest,
		name:     "ServerNameExtensionClientMismatch",
		config: Config{
			Bugs: ProtocolBugs{
				ExpectServerName: "mismatch.com",
			},
		},
		flags:              []string{"-host-name", "example.com"},
		shouldFail:         true,
		expectedLocalError: "tls: unexpected server name",
	})
	testCases = append(testCases, testCase{
		testType: clientTest,
		name:     "ServerNameExtensionClientMissing",
		config: Config{
			Bugs: ProtocolBugs{
				ExpectServerName: "missing.com",
			},
		},
		shouldFail:         true,
		expectedLocalError: "tls: unexpected server name",
	})
	testCases = append(testCases, testCase{
		testType: serverTest,
		name:     "ServerNameExtensionServer",
		config: Config{
			ServerName: "example.com",
		},
		flags:         []string{"-expect-server-name", "example.com"},
		resumeSession: true,
	})
	testCases = append(testCases, testCase{
		testType: clientTest,
		name:     "ALPNClient",
		config: Config{
			NextProtos: []string{"foo"},
		},
		flags: []string{
			"-advertise-alpn", "\x03foo\x03bar\x03baz",
			"-expect-alpn", "foo",
		},
		expectedNextProto:     "foo",
		expectedNextProtoType: alpn,
		resumeSession:         true,
	})
	testCases = append(testCases, testCase{
		testType: serverTest,
		name:     "ALPNServer",
		config: Config{
			NextProtos: []string{"foo", "bar", "baz"},
		},
		flags: []string{
			"-expect-advertised-alpn", "\x03foo\x03bar\x03baz",
			"-select-alpn", "foo",
		},
		expectedNextProto:     "foo",
		expectedNextProtoType: alpn,
		resumeSession:         true,
	})
	// Test that the server prefers ALPN over NPN.
	testCases = append(testCases, testCase{
		testType: serverTest,
		name:     "ALPNServer-Preferred",
		config: Config{
			NextProtos: []string{"foo", "bar", "baz"},
		},
		flags: []string{
			"-expect-advertised-alpn", "\x03foo\x03bar\x03baz",
			"-select-alpn", "foo",
			"-advertise-npn", "\x03foo\x03bar\x03baz",
		},
		expectedNextProto:     "foo",
		expectedNextProtoType: alpn,
		resumeSession:         true,
	})
	testCases = append(testCases, testCase{
		testType: serverTest,
		name:     "ALPNServer-Preferred-Swapped",
		config: Config{
			NextProtos: []string{"foo", "bar", "baz"},
			Bugs: ProtocolBugs{
				SwapNPNAndALPN: true,
			},
		},
		flags: []string{
			"-expect-advertised-alpn", "\x03foo\x03bar\x03baz",
			"-select-alpn", "foo",
			"-advertise-npn", "\x03foo\x03bar\x03baz",
		},
		expectedNextProto:     "foo",
		expectedNextProtoType: alpn,
		resumeSession:         true,
	})
	var emptyString string
	testCases = append(testCases, testCase{
		testType: clientTest,
		name:     "ALPNClient-EmptyProtocolName",
		config: Config{
			NextProtos: []string{""},
			Bugs: ProtocolBugs{
				// A server returning an empty ALPN protocol
				// should be rejected.
				ALPNProtocol: &emptyString,
			},
		},
		flags: []string{
			"-advertise-alpn", "\x03foo",
		},
		shouldFail:    true,
		expectedError: ":PARSE_TLSEXT:",
	})
	testCases = append(testCases, testCase{
		testType: serverTest,
		name:     "ALPNServer-EmptyProtocolName",
		config: Config{
			// A ClientHello containing an empty ALPN protocol
			// should be rejected.
			NextProtos: []string{"foo", "", "baz"},
		},
		flags: []string{
			"-select-alpn", "foo",
		},
		shouldFail:    true,
		expectedError: ":PARSE_TLSEXT:",
	})
	// Test that negotiating both NPN and ALPN is forbidden.
	testCases = append(testCases, testCase{
		name: "NegotiateALPNAndNPN",
		config: Config{
			NextProtos: []string{"foo", "bar", "baz"},
			Bugs: ProtocolBugs{
				NegotiateALPNAndNPN: true,
			},
		},
		flags: []string{
			"-advertise-alpn", "\x03foo",
			"-select-next-proto", "foo",
		},
		shouldFail:    true,
		expectedError: ":NEGOTIATED_BOTH_NPN_AND_ALPN:",
	})
	testCases = append(testCases, testCase{
		name: "NegotiateALPNAndNPN-Swapped",
		config: Config{
			NextProtos: []string{"foo", "bar", "baz"},
			Bugs: ProtocolBugs{
				NegotiateALPNAndNPN: true,
				SwapNPNAndALPN:      true,
			},
		},
		flags: []string{
			"-advertise-alpn", "\x03foo",
			"-select-next-proto", "foo",
		},
		shouldFail:    true,
		expectedError: ":NEGOTIATED_BOTH_NPN_AND_ALPN:",
	})
	// Resume with a corrupt ticket.
	testCases = append(testCases, testCase{
		testType: serverTest,
		name:     "CorruptTicket",
		config: Config{
			Bugs: ProtocolBugs{
				CorruptTicket: true,
			},
		},
		resumeSession:        true,
		expectResumeRejected: true,
	})
	// Test the ticket callback, with and without renewal.
	testCases = append(testCases, testCase{
		testType:      serverTest,
		name:          "TicketCallback",
		resumeSession: true,
		flags:         []string{"-use-ticket-callback"},
	})
	testCases = append(testCases, testCase{
		testType: serverTest,
		name:     "TicketCallback-Renew",
		config: Config{
			Bugs: ProtocolBugs{
				ExpectNewTicket: true,
			},
		},
		flags:         []string{"-use-ticket-callback", "-renew-ticket"},
		resumeSession: true,
	})
	// Resume with an oversized session id.
	testCases = append(testCases, testCase{
		testType: serverTest,
		name:     "OversizedSessionId",
		config: Config{
			Bugs: ProtocolBugs{
				OversizedSessionId: true,
			},
		},
		resumeSession: true,
		shouldFail:    true,
		expectedError: ":DECODE_ERROR:",
	})
	// Basic DTLS-SRTP tests. Include fake profiles to ensure they
	// are ignored.
	testCases = append(testCases, testCase{
		protocol: dtls,
		name:     "SRTP-Client",
		config: Config{
			SRTPProtectionProfiles: []uint16{40, SRTP_AES128_CM_HMAC_SHA1_80, 42},
		},
		flags: []string{
			"-srtp-profiles",
			"SRTP_AES128_CM_SHA1_80:SRTP_AES128_CM_SHA1_32",
		},
		expectedSRTPProtectionProfile: SRTP_AES128_CM_HMAC_SHA1_80,
	})
	testCases = append(testCases, testCase{
		protocol: dtls,
		testType: serverTest,
		name:     "SRTP-Server",
		config: Config{
			SRTPProtectionProfiles: []uint16{40, SRTP_AES128_CM_HMAC_SHA1_80, 42},
		},
		flags: []string{
			"-srtp-profiles",
			"SRTP_AES128_CM_SHA1_80:SRTP_AES128_CM_SHA1_32",
		},
		expectedSRTPProtectionProfile: SRTP_AES128_CM_HMAC_SHA1_80,
	})
	// Test that the MKI is ignored.
	testCases = append(testCases, testCase{
		protocol: dtls,
		testType: serverTest,
		name:     "SRTP-Server-IgnoreMKI",
		config: Config{
			SRTPProtectionProfiles: []uint16{SRTP_AES128_CM_HMAC_SHA1_80},
			Bugs: ProtocolBugs{
				SRTPMasterKeyIdentifer: "bogus",
			},
		},
		flags: []string{
			"-srtp-profiles",
			"SRTP_AES128_CM_SHA1_80:SRTP_AES128_CM_SHA1_32",
		},
		expectedSRTPProtectionProfile: SRTP_AES128_CM_HMAC_SHA1_80,
	})
	// Test that SRTP isn't negotiated on the server if there were
	// no matching profiles.
	testCases = append(testCases, testCase{
		protocol: dtls,
		testType: serverTest,
		name:     "SRTP-Server-NoMatch",
		config: Config{
			SRTPProtectionProfiles: []uint16{100, 101, 102},
		},
		flags: []string{
			"-srtp-profiles",
			"SRTP_AES128_CM_SHA1_80:SRTP_AES128_CM_SHA1_32",
		},
		expectedSRTPProtectionProfile: 0,
	})
	// Test that the server returning an invalid SRTP profile is
	// flagged as an error by the client.
	testCases = append(testCases, testCase{
		protocol: dtls,
		name:     "SRTP-Client-NoMatch",
		config: Config{
			Bugs: ProtocolBugs{
				SendSRTPProtectionProfile: SRTP_AES128_CM_HMAC_SHA1_32,
			},
		},
		flags: []string{
			"-srtp-profiles",
			"SRTP_AES128_CM_SHA1_80",
		},
		shouldFail:    true,
		expectedError: ":BAD_SRTP_PROTECTION_PROFILE_LIST:",
	})
	// Test SCT list.
	testCases = append(testCases, testCase{
		name:     "SignedCertificateTimestampList-Client",
		testType: clientTest,
		flags: []string{
			"-enable-signed-cert-timestamps",
			"-expect-signed-cert-timestamps",
			base64.StdEncoding.EncodeToString(testSCTList),
		},
		resumeSession: true,
	})
	testCases = append(testCases, testCase{
		name:     "SignedCertificateTimestampList-Server",
		testType: serverTest,
		flags: []string{
			"-signed-cert-timestamps",
			base64.StdEncoding.EncodeToString(testSCTList),
		},
		expectedSCTList: testSCTList,
		resumeSession:   true,
	})
	testCases = append(testCases, testCase{
		testType: clientTest,
		name:     "ClientHelloPadding",
		config: Config{
			Bugs: ProtocolBugs{
				RequireClientHelloSize: 512,
			},
		},
		// This hostname just needs to be long enough to push the
		// ClientHello into F5's danger zone between 256 and 511 bytes
		// long.
		flags: []string{"-host-name", "01234567890123456789012345678901234567890123456789012345678901234567890123456789.com"},
	})
}

func addResumptionVersionTests() {
	for _, sessionVers := range tlsVersions {
		for _, resumeVers := range tlsVersions {
			protocols := []protocol{tls}
			if sessionVers.hasDTLS && resumeVers.hasDTLS {
				protocols = append(protocols, dtls)
			}
			for _, protocol := range protocols {
				suffix := "-" + sessionVers.name + "-" + resumeVers.name
				if protocol == dtls {
					suffix += "-DTLS"
				}

				if sessionVers.version == resumeVers.version {
					testCases = append(testCases, testCase{
						protocol:      protocol,
						name:          "Resume-Client" + suffix,
						resumeSession: true,
						config: Config{
							MaxVersion:   sessionVers.version,
							CipherSuites: []uint16{TLS_RSA_WITH_AES_128_CBC_SHA},
						},
						expectedVersion:       sessionVers.version,
						expectedResumeVersion: resumeVers.version,
					})
				} else {
					testCases = append(testCases, testCase{
						protocol:      protocol,
						name:          "Resume-Client-Mismatch" + suffix,
						resumeSession: true,
						config: Config{
							MaxVersion:   sessionVers.version,
							CipherSuites: []uint16{TLS_RSA_WITH_AES_128_CBC_SHA},
						},
						expectedVersion: sessionVers.version,
						resumeConfig: &Config{
							MaxVersion:   resumeVers.version,
							CipherSuites: []uint16{TLS_RSA_WITH_AES_128_CBC_SHA},
							Bugs: ProtocolBugs{
								AllowSessionVersionMismatch: true,
							},
						},
						expectedResumeVersion: resumeVers.version,
						shouldFail:            true,
						expectedError:         ":OLD_SESSION_VERSION_NOT_RETURNED:",
					})
				}

				testCases = append(testCases, testCase{
					protocol:      protocol,
					name:          "Resume-Client-NoResume" + suffix,
					resumeSession: true,
					config: Config{
						MaxVersion:   sessionVers.version,
						CipherSuites: []uint16{TLS_RSA_WITH_AES_128_CBC_SHA},
					},
					expectedVersion: sessionVers.version,
					resumeConfig: &Config{
						MaxVersion:   resumeVers.version,
						CipherSuites: []uint16{TLS_RSA_WITH_AES_128_CBC_SHA},
					},
					newSessionsOnResume:   true,
					expectResumeRejected:  true,
					expectedResumeVersion: resumeVers.version,
				})

				testCases = append(testCases, testCase{
					protocol:      protocol,
					testType:      serverTest,
					name:          "Resume-Server" + suffix,
					resumeSession: true,
					config: Config{
						MaxVersion:   sessionVers.version,
						CipherSuites: []uint16{TLS_RSA_WITH_AES_128_CBC_SHA},
					},
					expectedVersion:      sessionVers.version,
					expectResumeRejected: sessionVers.version != resumeVers.version,
					resumeConfig: &Config{
						MaxVersion:   resumeVers.version,
						CipherSuites: []uint16{TLS_RSA_WITH_AES_128_CBC_SHA},
					},
					expectedResumeVersion: resumeVers.version,
				})
			}
		}
	}

	testCases = append(testCases, testCase{
		name:          "Resume-Client-CipherMismatch",
		resumeSession: true,
		config: Config{
			CipherSuites: []uint16{TLS_RSA_WITH_AES_128_GCM_SHA256},
		},
		resumeConfig: &Config{
			CipherSuites: []uint16{TLS_RSA_WITH_AES_128_GCM_SHA256},
			Bugs: ProtocolBugs{
				SendCipherSuite: TLS_RSA_WITH_AES_128_CBC_SHA,
			},
		},
		shouldFail:    true,
		expectedError: ":OLD_SESSION_CIPHER_NOT_RETURNED:",
	})
}

func addRenegotiationTests() {
	// Servers cannot renegotiate.
	testCases = append(testCases, testCase{
		testType:           serverTest,
		name:               "Renegotiate-Server-Forbidden",
		renegotiate:        true,
		flags:              []string{"-reject-peer-renegotiations"},
		shouldFail:         true,
		expectedError:      ":NO_RENEGOTIATION:",
		expectedLocalError: "remote error: no renegotiation",
	})
	// The server shouldn't echo the renegotiation extension unless
	// requested by the client.
	testCases = append(testCases, testCase{
		testType: serverTest,
		name:     "Renegotiate-Server-NoExt",
		config: Config{
			Bugs: ProtocolBugs{
				NoRenegotiationInfo:      true,
				RequireRenegotiationInfo: true,
			},
		},
		shouldFail:         true,
		expectedLocalError: "renegotiation extension missing",
	})
	// The renegotiation SCSV should be sufficient for the server to echo
	// the extension.
	testCases = append(testCases, testCase{
		testType: serverTest,
		name:     "Renegotiate-Server-NoExt-SCSV",
		config: Config{
			Bugs: ProtocolBugs{
				NoRenegotiationInfo:      true,
				SendRenegotiationSCSV:    true,
				RequireRenegotiationInfo: true,
			},
		},
	})
	testCases = append(testCases, testCase{
		name: "Renegotiate-Client",
		config: Config{
			Bugs: ProtocolBugs{
				FailIfResumeOnRenego: true,
			},
		},
		renegotiate: true,
	})
	testCases = append(testCases, testCase{
		name:        "Renegotiate-Client-EmptyExt",
		renegotiate: true,
		config: Config{
			Bugs: ProtocolBugs{
				EmptyRenegotiationInfo: true,
			},
		},
		shouldFail:    true,
		expectedError: ":RENEGOTIATION_MISMATCH:",
	})
	testCases = append(testCases, testCase{
		name:        "Renegotiate-Client-BadExt",
		renegotiate: true,
		config: Config{
			Bugs: ProtocolBugs{
				BadRenegotiationInfo: true,
			},
		},
		shouldFail:    true,
		expectedError: ":RENEGOTIATION_MISMATCH:",
	})
	testCases = append(testCases, testCase{
		name: "Renegotiate-Client-NoExt",
		config: Config{
			Bugs: ProtocolBugs{
				NoRenegotiationInfo: true,
			},
		},
		shouldFail:    true,
		expectedError: ":UNSAFE_LEGACY_RENEGOTIATION_DISABLED:",
		flags:         []string{"-no-legacy-server-connect"},
	})
	testCases = append(testCases, testCase{
		name:        "Renegotiate-Client-NoExt-Allowed",
		renegotiate: true,
		config: Config{
			Bugs: ProtocolBugs{
				NoRenegotiationInfo: true,
			},
		},
	})
	testCases = append(testCases, testCase{
		name:        "Renegotiate-Client-SwitchCiphers",
		renegotiate: true,
		config: Config{
			CipherSuites: []uint16{TLS_RSA_WITH_RC4_128_SHA},
		},
		renegotiateCiphers: []uint16{TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256},
	})
	testCases = append(testCases, testCase{
		name:        "Renegotiate-Client-SwitchCiphers2",
		renegotiate: true,
		config: Config{
			CipherSuites: []uint16{TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256},
		},
		renegotiateCiphers: []uint16{TLS_RSA_WITH_RC4_128_SHA},
	})
	testCases = append(testCases, testCase{
		name:               "Renegotiate-Client-Forbidden",
		renegotiate:        true,
		flags:              []string{"-reject-peer-renegotiations"},
		shouldFail:         true,
		expectedError:      ":NO_RENEGOTIATION:",
		expectedLocalError: "remote error: no renegotiation",
	})
	testCases = append(testCases, testCase{
		name:        "Renegotiate-SameClientVersion",
		renegotiate: true,
		config: Config{
			MaxVersion: VersionTLS10,
			Bugs: ProtocolBugs{
				RequireSameRenegoClientVersion: true,
			},
		},
	})
	testCases = append(testCases, testCase{
		name:        "Renegotiate-FalseStart",
		renegotiate: true,
		config: Config{
			CipherSuites: []uint16{TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256},
			NextProtos:   []string{"foo"},
		},
		flags: []string{
			"-false-start",
			"-select-next-proto", "foo",
		},
		shimWritesFirst: true,
	})
}

func addDTLSReplayTests() {
	// Test that sequence number replays are detected.
	testCases = append(testCases, testCase{
		protocol:     dtls,
		name:         "DTLS-Replay",
		messageCount: 200,
		replayWrites: true,
	})

	// Test the incoming sequence number skipping by values larger
	// than the retransmit window.
	testCases = append(testCases, testCase{
		protocol: dtls,
		name:     "DTLS-Replay-LargeGaps",
		config: Config{
			Bugs: ProtocolBugs{
				SequenceNumberMapping: func(in uint64) uint64 {
					return in * 127
				},
			},
		},
		messageCount: 200,
		replayWrites: true,
	})

	// Test the incoming sequence number changing non-monotonically.
	testCases = append(testCases, testCase{
		protocol: dtls,
		name:     "DTLS-Replay-NonMonotonic",
		config: Config{
			Bugs: ProtocolBugs{
				SequenceNumberMapping: func(in uint64) uint64 {
					return in ^ 31
				},
			},
		},
		messageCount: 200,
		replayWrites: true,
	})
}

var testHashes = []struct {
	name string
	id   uint8
}{
	{"SHA1", hashSHA1},
	{"SHA224", hashSHA224},
	{"SHA256", hashSHA256},
	{"SHA384", hashSHA384},
	{"SHA512", hashSHA512},
}

func addSigningHashTests() {
	// Make sure each hash works. Include some fake hashes in the list and
	// ensure they're ignored.
	for _, hash := range testHashes {
		testCases = append(testCases, testCase{
			name: "SigningHash-ClientAuth-" + hash.name,
			config: Config{
				ClientAuth: RequireAnyClientCert,
				SignatureAndHashes: []signatureAndHash{
					{signatureRSA, 42},
					{signatureRSA, hash.id},
					{signatureRSA, 255},
				},
			},
			flags: []string{
				"-cert-file", path.Join(*resourceDir, rsaCertificateFile),
				"-key-file", path.Join(*resourceDir, rsaKeyFile),
			},
		})

		testCases = append(testCases, testCase{
			testType: serverTest,
			name:     "SigningHash-ServerKeyExchange-Sign-" + hash.name,
			config: Config{
				CipherSuites: []uint16{TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256},
				SignatureAndHashes: []signatureAndHash{
					{signatureRSA, 42},
					{signatureRSA, hash.id},
					{signatureRSA, 255},
				},
			},
		})
	}

	// Test that hash resolution takes the signature type into account.
	testCases = append(testCases, testCase{
		name: "SigningHash-ClientAuth-SignatureType",
		config: Config{
			ClientAuth: RequireAnyClientCert,
			SignatureAndHashes: []signatureAndHash{
				{signatureECDSA, hashSHA512},
				{signatureRSA, hashSHA384},
				{signatureECDSA, hashSHA1},
			},
		},
		flags: []string{
			"-cert-file", path.Join(*resourceDir, rsaCertificateFile),
			"-key-file", path.Join(*resourceDir, rsaKeyFile),
		},
	})

	testCases = append(testCases, testCase{
		testType: serverTest,
		name:     "SigningHash-ServerKeyExchange-SignatureType",
		config: Config{
			CipherSuites: []uint16{TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256},
			SignatureAndHashes: []signatureAndHash{
				{signatureECDSA, hashSHA512},
				{signatureRSA, hashSHA384},
				{signatureECDSA, hashSHA1},
			},
		},
	})

	// Test that, if the list is missing, the peer falls back to SHA-1.
	testCases = append(testCases, testCase{
		name: "SigningHash-ClientAuth-Fallback",
		config: Config{
			ClientAuth: RequireAnyClientCert,
			SignatureAndHashes: []signatureAndHash{
				{signatureRSA, hashSHA1},
			},
			Bugs: ProtocolBugs{
				NoSignatureAndHashes: true,
			},
		},
		flags: []string{
			"-cert-file", path.Join(*resourceDir, rsaCertificateFile),
			"-key-file", path.Join(*resourceDir, rsaKeyFile),
		},
	})

	testCases = append(testCases, testCase{
		testType: serverTest,
		name:     "SigningHash-ServerKeyExchange-Fallback",
		config: Config{
			CipherSuites: []uint16{TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256},
			SignatureAndHashes: []signatureAndHash{
				{signatureRSA, hashSHA1},
			},
			Bugs: ProtocolBugs{
				NoSignatureAndHashes: true,
			},
		},
	})

	// Test that hash preferences are enforced. BoringSSL defaults to
	// rejecting MD5 signatures.
	testCases = append(testCases, testCase{
		testType: serverTest,
		name:     "SigningHash-ClientAuth-Enforced",
		config: Config{
			Certificates: []Certificate{rsaCertificate},
			SignatureAndHashes: []signatureAndHash{
				{signatureRSA, hashMD5},
				// Advertise SHA-1 so the handshake will
				// proceed, but the shim's preferences will be
				// ignored in CertificateVerify generation, so
				// MD5 will be chosen.
				{signatureRSA, hashSHA1},
			},
			Bugs: ProtocolBugs{
				IgnorePeerSignatureAlgorithmPreferences: true,
			},
		},
		flags:         []string{"-require-any-client-certificate"},
		shouldFail:    true,
		expectedError: ":WRONG_SIGNATURE_TYPE:",
	})

	testCases = append(testCases, testCase{
		name: "SigningHash-ServerKeyExchange-Enforced",
		config: Config{
			CipherSuites: []uint16{TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256},
			SignatureAndHashes: []signatureAndHash{
				{signatureRSA, hashMD5},
			},
			Bugs: ProtocolBugs{
				IgnorePeerSignatureAlgorithmPreferences: true,
			},
		},
		shouldFail:    true,
		expectedError: ":WRONG_SIGNATURE_TYPE:",
	})

	// Test that the agreed upon digest respects the client preferences and
	// the server digests.
	testCases = append(testCases, testCase{
		name: "Agree-Digest-Fallback",
		config: Config{
			ClientAuth: RequireAnyClientCert,
			SignatureAndHashes: []signatureAndHash{
				{signatureRSA, hashSHA512},
				{signatureRSA, hashSHA1},
			},
		},
		flags: []string{
			"-cert-file", path.Join(*resourceDir, rsaCertificateFile),
			"-key-file", path.Join(*resourceDir, rsaKeyFile),
		},
		digestPrefs:                     "SHA256",
		expectedClientCertSignatureHash: hashSHA1,
	})
	testCases = append(testCases, testCase{
		name: "Agree-Digest-SHA256",
		config: Config{
			ClientAuth: RequireAnyClientCert,
			SignatureAndHashes: []signatureAndHash{
				{signatureRSA, hashSHA1},
				{signatureRSA, hashSHA256},
			},
		},
		flags: []string{
			"-cert-file", path.Join(*resourceDir, rsaCertificateFile),
			"-key-file", path.Join(*resourceDir, rsaKeyFile),
		},
		digestPrefs:                     "SHA256,SHA1",
		expectedClientCertSignatureHash: hashSHA256,
	})
	testCases = append(testCases, testCase{
		name: "Agree-Digest-SHA1",
		config: Config{
			ClientAuth: RequireAnyClientCert,
			SignatureAndHashes: []signatureAndHash{
				{signatureRSA, hashSHA1},
			},
		},
		flags: []string{
			"-cert-file", path.Join(*resourceDir, rsaCertificateFile),
			"-key-file", path.Join(*resourceDir, rsaKeyFile),
		},
		digestPrefs:                     "SHA512,SHA256,SHA1",
		expectedClientCertSignatureHash: hashSHA1,
	})
	testCases = append(testCases, testCase{
		name: "Agree-Digest-Default",
		config: Config{
			ClientAuth: RequireAnyClientCert,
			SignatureAndHashes: []signatureAndHash{
				{signatureRSA, hashSHA256},
				{signatureECDSA, hashSHA256},
				{signatureRSA, hashSHA1},
				{signatureECDSA, hashSHA1},
			},
		},
		flags: []string{
			"-cert-file", path.Join(*resourceDir, rsaCertificateFile),
			"-key-file", path.Join(*resourceDir, rsaKeyFile),
		},
		expectedClientCertSignatureHash: hashSHA256,
	})
}

// timeouts is the retransmit schedule for BoringSSL. It doubles and
// caps at 60 seconds. On the 13th timeout, it gives up.
var timeouts = []time.Duration{
	1 * time.Second,
	2 * time.Second,
	4 * time.Second,
	8 * time.Second,
	16 * time.Second,
	32 * time.Second,
	60 * time.Second,
	60 * time.Second,
	60 * time.Second,
	60 * time.Second,
	60 * time.Second,
	60 * time.Second,
	60 * time.Second,
}

func addDTLSRetransmitTests() {
	// Test that this is indeed the timeout schedule. Stress all
	// four patterns of handshake.
	for i := 1; i < len(timeouts); i++ {
		number := strconv.Itoa(i)
		testCases = append(testCases, testCase{
			protocol: dtls,
			name:     "DTLS-Retransmit-Client-" + number,
			config: Config{
				Bugs: ProtocolBugs{
					TimeoutSchedule: timeouts[:i],
				},
			},
			resumeSession: true,
			flags:         []string{"-async"},
		})
		testCases = append(testCases, testCase{
			protocol: dtls,
			testType: serverTest,
			name:     "DTLS-Retransmit-Server-" + number,
			config: Config{
				Bugs: ProtocolBugs{
					TimeoutSchedule: timeouts[:i],
				},
			},
			resumeSession: true,
			flags:         []string{"-async"},
		})
	}

	// Test that exceeding the timeout schedule hits a read
	// timeout.
	testCases = append(testCases, testCase{
		protocol: dtls,
		name:     "DTLS-Retransmit-Timeout",
		config: Config{
			Bugs: ProtocolBugs{
				TimeoutSchedule: timeouts,
			},
		},
		resumeSession: true,
		flags:         []string{"-async"},
		shouldFail:    true,
		expectedError: ":READ_TIMEOUT_EXPIRED:",
	})

	// Test that timeout handling has a fudge factor, due to API
	// problems.
	testCases = append(testCases, testCase{
		protocol: dtls,
		name:     "DTLS-Retransmit-Fudge",
		config: Config{
			Bugs: ProtocolBugs{
				TimeoutSchedule: []time.Duration{
					timeouts[0] - 10*time.Millisecond,
				},
			},
		},
		resumeSession: true,
		flags:         []string{"-async"},
	})

	// Test that the final Finished retransmitting isn't
	// duplicated if the peer badly fragments everything.
	testCases = append(testCases, testCase{
		testType: serverTest,
		protocol: dtls,
		name:     "DTLS-Retransmit-Fragmented",
		config: Config{
			Bugs: ProtocolBugs{
				TimeoutSchedule:          []time.Duration{timeouts[0]},
				MaxHandshakeRecordLength: 2,
			},
		},
		flags: []string{"-async"},
	})
}

func addExportKeyingMaterialTests() {
	for _, vers := range tlsVersions {
		if vers.version == VersionSSL30 {
			continue
		}
		testCases = append(testCases, testCase{
			name: "ExportKeyingMaterial-" + vers.name,
			config: Config{
				MaxVersion: vers.version,
			},
			exportKeyingMaterial: 1024,
			exportLabel:          "label",
			exportContext:        "context",
			useExportContext:     true,
		})
		testCases = append(testCases, testCase{
			name: "ExportKeyingMaterial-NoContext-" + vers.name,
			config: Config{
				MaxVersion: vers.version,
			},
			exportKeyingMaterial: 1024,
		})
		testCases = append(testCases, testCase{
			name: "ExportKeyingMaterial-EmptyContext-" + vers.name,
			config: Config{
				MaxVersion: vers.version,
			},
			exportKeyingMaterial: 1024,
			useExportContext:     true,
		})
		testCases = append(testCases, testCase{
			name: "ExportKeyingMaterial-Small-" + vers.name,
			config: Config{
				MaxVersion: vers.version,
			},
			exportKeyingMaterial: 1,
			exportLabel:          "label",
			exportContext:        "context",
			useExportContext:     true,
		})
	}
	testCases = append(testCases, testCase{
		name: "ExportKeyingMaterial-SSL3",
		config: Config{
			MaxVersion: VersionSSL30,
		},
		exportKeyingMaterial: 1024,
		exportLabel:          "label",
		exportContext:        "context",
		useExportContext:     true,
		shouldFail:           true,
		expectedError:        "failed to export keying material",
	})
}

func addTLSUniqueTests() {
	for _, isClient := range []bool{false, true} {
		for _, isResumption := range []bool{false, true} {
			for _, hasEMS := range []bool{false, true} {
				var suffix string
				if isResumption {
					suffix = "Resume-"
				} else {
					suffix = "Full-"
				}

				if hasEMS {
					suffix += "EMS-"
				} else {
					suffix += "NoEMS-"
				}

				if isClient {
					suffix += "Client"
				} else {
					suffix += "Server"
				}

				test := testCase{
					name:          "TLSUnique-" + suffix,
					testTLSUnique: true,
					config: Config{
						Bugs: ProtocolBugs{
							NoExtendedMasterSecret: !hasEMS,
						},
					},
				}

				if isResumption {
					test.resumeSession = true
					test.resumeConfig = &Config{
						Bugs: ProtocolBugs{
							NoExtendedMasterSecret: !hasEMS,
						},
					}
				}

				if isResumption && !hasEMS {
					test.shouldFail = true
					test.expectedError = "failed to get tls-unique"
				}

				testCases = append(testCases, test)
			}
		}
	}
}

func addCustomExtensionTests() {
	expectedContents := "custom extension"
	emptyString := ""

	for _, isClient := range []bool{false, true} {
		suffix := "Server"
		flag := "-enable-server-custom-extension"
		testType := serverTest
		if isClient {
			suffix = "Client"
			flag = "-enable-client-custom-extension"
			testType = clientTest
		}

		testCases = append(testCases, testCase{
			testType: testType,
			name:     "CustomExtensions-" + suffix,
			config: Config{
				Bugs: ProtocolBugs{
					CustomExtension:         expectedContents,
					ExpectedCustomExtension: &expectedContents,
				},
			},
			flags: []string{flag},
		})

		// If the parse callback fails, the handshake should also fail.
		testCases = append(testCases, testCase{
			testType: testType,
			name:     "CustomExtensions-ParseError-" + suffix,
			config: Config{
				Bugs: ProtocolBugs{
					CustomExtension:         expectedContents + "foo",
					ExpectedCustomExtension: &expectedContents,
				},
			},
			flags:         []string{flag},
			shouldFail:    true,
			expectedError: ":CUSTOM_EXTENSION_ERROR:",
		})

		// If the add callback fails, the handshake should also fail.
		testCases = append(testCases, testCase{
			testType: testType,
			name:     "CustomExtensions-FailAdd-" + suffix,
			config: Config{
				Bugs: ProtocolBugs{
					CustomExtension:         expectedContents,
					ExpectedCustomExtension: &expectedContents,
				},
			},
			flags:         []string{flag, "-custom-extension-fail-add"},
			shouldFail:    true,
			expectedError: ":CUSTOM_EXTENSION_ERROR:",
		})

		// If the add callback returns zero, no extension should be
		// added.
		skipCustomExtension := expectedContents
		if isClient {
			// For the case where the client skips sending the
			// custom extension, the server must not “echo” it.
			skipCustomExtension = ""
		}
		testCases = append(testCases, testCase{
			testType: testType,
			name:     "CustomExtensions-Skip-" + suffix,
			config: Config{
				Bugs: ProtocolBugs{
					CustomExtension:         skipCustomExtension,
					ExpectedCustomExtension: &emptyString,
				},
			},
			flags: []string{flag, "-custom-extension-skip"},
		})
	}

	// The custom extension add callback should not be called if the client
	// doesn't send the extension.
	testCases = append(testCases, testCase{
		testType: serverTest,
		name:     "CustomExtensions-NotCalled-Server",
		config: Config{
			Bugs: ProtocolBugs{
				ExpectedCustomExtension: &emptyString,
			},
		},
		flags: []string{"-enable-server-custom-extension", "-custom-extension-fail-add"},
	})

	// Test an unknown extension from the server.
	testCases = append(testCases, testCase{
		testType: clientTest,
		name:     "UnknownExtension-Client",
		config: Config{
			Bugs: ProtocolBugs{
				CustomExtension: expectedContents,
			},
		},
		shouldFail:    true,
		expectedError: ":UNEXPECTED_EXTENSION:",
	})
}

func worker(statusChan chan statusMsg, c chan *testCase, shimPath string, wg *sync.WaitGroup) {
	defer wg.Done()

	for test := range c {
		var err error

		if *mallocTest < 0 {
			statusChan <- statusMsg{test: test, started: true}
			err = runTest(test, shimPath, -1)
		} else {
			for mallocNumToFail := int64(*mallocTest); ; mallocNumToFail++ {
				statusChan <- statusMsg{test: test, started: true}
				if err = runTest(test, shimPath, mallocNumToFail); err != errMoreMallocs {
					if err != nil {
						fmt.Printf("\n\nmalloc test failed at %d: %s\n", mallocNumToFail, err)
					}
					break
				}
			}
		}
		statusChan <- statusMsg{test: test, err: err}
	}
}

type statusMsg struct {
	test    *testCase
	started bool
	err     error
}

func statusPrinter(doneChan chan *testOutput, statusChan chan statusMsg, total int) {
	var started, done, failed, lineLen int

	testOutput := newTestOutput()
	for msg := range statusChan {
		if !*pipe {
			// Erase the previous status line.
			var erase string
			for i := 0; i < lineLen; i++ {
				erase += "\b \b"
			}
			fmt.Print(erase)
		}

		if msg.started {
			started++
		} else {
			done++

			if msg.err != nil {
				fmt.Printf("FAILED (%s)\n%s\n", msg.test.name, msg.err)
				failed++
				testOutput.addResult(msg.test.name, "FAIL")
			} else {
				if *pipe {
					// Print each test instead of a status line.
					fmt.Printf("PASSED (%s)\n", msg.test.name)
				}
				testOutput.addResult(msg.test.name, "PASS")
			}
		}

		if !*pipe {
			// Print a new status line.
			line := fmt.Sprintf("%d/%d/%d/%d", failed, done, started, total)
			lineLen = len(line)
			os.Stdout.WriteString(line)
		}
	}

	doneChan <- testOutput
}

func main() {
	flag.Parse()
	*resourceDir = path.Clean(*resourceDir)

	addBasicTests()
	addCipherSuiteTests()
	addBadECDSASignatureTests()
	addCBCPaddingTests()
	addCBCSplittingTests()
	addClientAuthTests()
	addDDoSCallbackTests()
	addVersionNegotiationTests()
	addMinimumVersionTests()
	addD5BugTests()
	addExtensionTests()
	addResumptionVersionTests()
	addExtendedMasterSecretTests()
	addRenegotiationTests()
	addDTLSReplayTests()
	addSigningHashTests()
	addDTLSRetransmitTests()
	addExportKeyingMaterialTests()
	addTLSUniqueTests()
	addCustomExtensionTests()
	for _, async := range []bool{false, true} {
		for _, splitHandshake := range []bool{false, true} {
			for _, protocol := range []protocol{tls, dtls} {
				addStateMachineCoverageTests(async, splitHandshake, protocol)
			}
		}
	}

	var wg sync.WaitGroup

	statusChan := make(chan statusMsg, *numWorkers)
	testChan := make(chan *testCase, *numWorkers)
	doneChan := make(chan *testOutput)

	go statusPrinter(doneChan, statusChan, len(testCases))

	for i := 0; i < *numWorkers; i++ {
		wg.Add(1)
		go worker(statusChan, testChan, *shimPath, &wg)
	}

	for i := range testCases {
		if len(*testToRun) == 0 || *testToRun == testCases[i].name {
			testChan <- &testCases[i]
		}
	}

	close(testChan)
	wg.Wait()
	close(statusChan)
	testOutput := <-doneChan

	fmt.Printf("\n")

	if *jsonOutput != "" {
		if err := testOutput.writeTo(*jsonOutput); err != nil {
			fmt.Fprintf(os.Stderr, "Error: %s\n", err)
		}
	}

	if !testOutput.allPassed {
		os.Exit(1)
	}
}
