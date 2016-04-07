// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runner

import (
	"container/list"
	"crypto"
	"crypto/ecdsa"
	"crypto/rand"
	"crypto/x509"
	"fmt"
	"io"
	"math/big"
	"strings"
	"sync"
	"time"
)

const (
	VersionSSL30 = 0x0300
	VersionTLS10 = 0x0301
	VersionTLS11 = 0x0302
	VersionTLS12 = 0x0303
)

const (
	maxPlaintext        = 16384        // maximum plaintext payload length
	maxCiphertext       = 16384 + 2048 // maximum ciphertext payload length
	tlsRecordHeaderLen  = 5            // record header length
	dtlsRecordHeaderLen = 13
	maxHandshake        = 65536 // maximum handshake we support (protocol max is 16 MB)

	minVersion = VersionSSL30
	maxVersion = VersionTLS12
)

// TLS record types.
type recordType uint8

const (
	recordTypeChangeCipherSpec recordType = 20
	recordTypeAlert            recordType = 21
	recordTypeHandshake        recordType = 22
	recordTypeApplicationData  recordType = 23
)

// TLS handshake message types.
const (
	typeHelloRequest        uint8 = 0
	typeClientHello         uint8 = 1
	typeServerHello         uint8 = 2
	typeHelloVerifyRequest  uint8 = 3
	typeNewSessionTicket    uint8 = 4
	typeCertificate         uint8 = 11
	typeServerKeyExchange   uint8 = 12
	typeCertificateRequest  uint8 = 13
	typeServerHelloDone     uint8 = 14
	typeCertificateVerify   uint8 = 15
	typeClientKeyExchange   uint8 = 16
	typeFinished            uint8 = 20
	typeCertificateStatus   uint8 = 22
	typeNextProtocol        uint8 = 67  // Not IANA assigned
	typeEncryptedExtensions uint8 = 203 // Not IANA assigned
)

// TLS compression types.
const (
	compressionNone uint8 = 0
)

// TLS extension numbers
const (
	extensionServerName                 uint16 = 0
	extensionStatusRequest              uint16 = 5
	extensionSupportedCurves            uint16 = 10
	extensionSupportedPoints            uint16 = 11
	extensionSignatureAlgorithms        uint16 = 13
	extensionUseSRTP                    uint16 = 14
	extensionALPN                       uint16 = 16
	extensionSignedCertificateTimestamp uint16 = 18
	extensionExtendedMasterSecret       uint16 = 23
	extensionSessionTicket              uint16 = 35
	extensionCustom                     uint16 = 1234  // not IANA assigned
	extensionNextProtoNeg               uint16 = 13172 // not IANA assigned
	extensionRenegotiationInfo          uint16 = 0xff01
	extensionChannelID                  uint16 = 30032 // not IANA assigned
)

// TLS signaling cipher suite values
const (
	scsvRenegotiation uint16 = 0x00ff
)

// CurveID is the type of a TLS identifier for an elliptic curve. See
// http://www.iana.org/assignments/tls-parameters/tls-parameters.xml#tls-parameters-8
type CurveID uint16

const (
	CurveP224 CurveID = 21
	CurveP256 CurveID = 23
	CurveP384 CurveID = 24
	CurveP521 CurveID = 25
)

// TLS Elliptic Curve Point Formats
// http://www.iana.org/assignments/tls-parameters/tls-parameters.xml#tls-parameters-9
const (
	pointFormatUncompressed uint8 = 0
)

// TLS CertificateStatusType (RFC 3546)
const (
	statusTypeOCSP uint8 = 1
)

// Certificate types (for certificateRequestMsg)
const (
	CertTypeRSASign    = 1 // A certificate containing an RSA key
	CertTypeDSSSign    = 2 // A certificate containing a DSA key
	CertTypeRSAFixedDH = 3 // A certificate containing a static DH key
	CertTypeDSSFixedDH = 4 // A certificate containing a static DH key

	// See RFC4492 sections 3 and 5.5.
	CertTypeECDSASign      = 64 // A certificate containing an ECDSA-capable public key, signed with ECDSA.
	CertTypeRSAFixedECDH   = 65 // A certificate containing an ECDH-capable public key, signed with RSA.
	CertTypeECDSAFixedECDH = 66 // A certificate containing an ECDH-capable public key, signed with ECDSA.

	// Rest of these are reserved by the TLS spec
)

// Hash functions for TLS 1.2 (See RFC 5246, section A.4.1)
const (
	hashMD5    uint8 = 1
	hashSHA1   uint8 = 2
	hashSHA224 uint8 = 3
	hashSHA256 uint8 = 4
	hashSHA384 uint8 = 5
	hashSHA512 uint8 = 6
)

// Signature algorithms for TLS 1.2 (See RFC 5246, section A.4.1)
const (
	signatureRSA   uint8 = 1
	signatureECDSA uint8 = 3
)

// signatureAndHash mirrors the TLS 1.2, SignatureAndHashAlgorithm struct. See
// RFC 5246, section A.4.1.
type signatureAndHash struct {
	signature, hash uint8
}

// supportedSKXSignatureAlgorithms contains the signature and hash algorithms
// that the code advertises as supported in a TLS 1.2 ClientHello.
var supportedSKXSignatureAlgorithms = []signatureAndHash{
	{signatureRSA, hashSHA256},
	{signatureECDSA, hashSHA256},
	{signatureRSA, hashSHA1},
	{signatureECDSA, hashSHA1},
}

// supportedClientCertSignatureAlgorithms contains the signature and hash
// algorithms that the code advertises as supported in a TLS 1.2
// CertificateRequest.
var supportedClientCertSignatureAlgorithms = []signatureAndHash{
	{signatureRSA, hashSHA256},
	{signatureECDSA, hashSHA256},
}

// SRTP protection profiles (See RFC 5764, section 4.1.2)
const (
	SRTP_AES128_CM_HMAC_SHA1_80 uint16 = 0x0001
	SRTP_AES128_CM_HMAC_SHA1_32        = 0x0002
)

// ConnectionState records basic TLS details about the connection.
type ConnectionState struct {
	Version                    uint16                // TLS version used by the connection (e.g. VersionTLS12)
	HandshakeComplete          bool                  // TLS handshake is complete
	DidResume                  bool                  // connection resumes a previous TLS connection
	CipherSuite                uint16                // cipher suite in use (TLS_RSA_WITH_RC4_128_SHA, ...)
	NegotiatedProtocol         string                // negotiated next protocol (from Config.NextProtos)
	NegotiatedProtocolIsMutual bool                  // negotiated protocol was advertised by server
	NegotiatedProtocolFromALPN bool                  // protocol negotiated with ALPN
	ServerName                 string                // server name requested by client, if any (server side only)
	PeerCertificates           []*x509.Certificate   // certificate chain presented by remote peer
	VerifiedChains             [][]*x509.Certificate // verified chains built from PeerCertificates
	ChannelID                  *ecdsa.PublicKey      // the channel ID for this connection
	SRTPProtectionProfile      uint16                // the negotiated DTLS-SRTP protection profile
	TLSUnique                  []byte                // the tls-unique channel binding
	SCTList                    []byte                // signed certificate timestamp list
	ClientCertSignatureHash    uint8                 // TLS id of the hash used by the client to sign the handshake
}

// ClientAuthType declares the policy the server will follow for
// TLS Client Authentication.
type ClientAuthType int

const (
	NoClientCert ClientAuthType = iota
	RequestClientCert
	RequireAnyClientCert
	VerifyClientCertIfGiven
	RequireAndVerifyClientCert
)

// ClientSessionState contains the state needed by clients to resume TLS
// sessions.
type ClientSessionState struct {
	sessionId            []uint8             // Session ID supplied by the server. nil if the session has a ticket.
	sessionTicket        []uint8             // Encrypted ticket used for session resumption with server
	vers                 uint16              // SSL/TLS version negotiated for the session
	cipherSuite          uint16              // Ciphersuite negotiated for the session
	masterSecret         []byte              // MasterSecret generated by client on a full handshake
	handshakeHash        []byte              // Handshake hash for Channel ID purposes.
	serverCertificates   []*x509.Certificate // Certificate chain presented by the server
	extendedMasterSecret bool                // Whether an extended master secret was used to generate the session
	sctList              []byte
	ocspResponse         []byte
}

// ClientSessionCache is a cache of ClientSessionState objects that can be used
// by a client to resume a TLS session with a given server. ClientSessionCache
// implementations should expect to be called concurrently from different
// goroutines.
type ClientSessionCache interface {
	// Get searches for a ClientSessionState associated with the given key.
	// On return, ok is true if one was found.
	Get(sessionKey string) (session *ClientSessionState, ok bool)

	// Put adds the ClientSessionState to the cache with the given key.
	Put(sessionKey string, cs *ClientSessionState)
}

// ServerSessionCache is a cache of sessionState objects that can be used by a
// client to resume a TLS session with a given server. ServerSessionCache
// implementations should expect to be called concurrently from different
// goroutines.
type ServerSessionCache interface {
	// Get searches for a sessionState associated with the given session
	// ID. On return, ok is true if one was found.
	Get(sessionId string) (session *sessionState, ok bool)

	// Put adds the sessionState to the cache with the given session ID.
	Put(sessionId string, session *sessionState)
}

// A Config structure is used to configure a TLS client or server.
// After one has been passed to a TLS function it must not be
// modified. A Config may be reused; the tls package will also not
// modify it.
type Config struct {
	// Rand provides the source of entropy for nonces and RSA blinding.
	// If Rand is nil, TLS uses the cryptographic random reader in package
	// crypto/rand.
	// The Reader must be safe for use by multiple goroutines.
	Rand io.Reader

	// Time returns the current time as the number of seconds since the epoch.
	// If Time is nil, TLS uses time.Now.
	Time func() time.Time

	// Certificates contains one or more certificate chains
	// to present to the other side of the connection.
	// Server configurations must include at least one certificate.
	Certificates []Certificate

	// NameToCertificate maps from a certificate name to an element of
	// Certificates. Note that a certificate name can be of the form
	// '*.example.com' and so doesn't have to be a domain name as such.
	// See Config.BuildNameToCertificate
	// The nil value causes the first element of Certificates to be used
	// for all connections.
	NameToCertificate map[string]*Certificate

	// RootCAs defines the set of root certificate authorities
	// that clients use when verifying server certificates.
	// If RootCAs is nil, TLS uses the host's root CA set.
	RootCAs *x509.CertPool

	// NextProtos is a list of supported, application level protocols.
	NextProtos []string

	// ServerName is used to verify the hostname on the returned
	// certificates unless InsecureSkipVerify is given. It is also included
	// in the client's handshake to support virtual hosting.
	ServerName string

	// ClientAuth determines the server's policy for
	// TLS Client Authentication. The default is NoClientCert.
	ClientAuth ClientAuthType

	// ClientCAs defines the set of root certificate authorities
	// that servers use if required to verify a client certificate
	// by the policy in ClientAuth.
	ClientCAs *x509.CertPool

	// ClientCertificateTypes defines the set of allowed client certificate
	// types. The default is CertTypeRSASign and CertTypeECDSASign.
	ClientCertificateTypes []byte

	// InsecureSkipVerify controls whether a client verifies the
	// server's certificate chain and host name.
	// If InsecureSkipVerify is true, TLS accepts any certificate
	// presented by the server and any host name in that certificate.
	// In this mode, TLS is susceptible to man-in-the-middle attacks.
	// This should be used only for testing.
	InsecureSkipVerify bool

	// CipherSuites is a list of supported cipher suites. If CipherSuites
	// is nil, TLS uses a list of suites supported by the implementation.
	CipherSuites []uint16

	// PreferServerCipherSuites controls whether the server selects the
	// client's most preferred ciphersuite, or the server's most preferred
	// ciphersuite. If true then the server's preference, as expressed in
	// the order of elements in CipherSuites, is used.
	PreferServerCipherSuites bool

	// SessionTicketsDisabled may be set to true to disable session ticket
	// (resumption) support.
	SessionTicketsDisabled bool

	// SessionTicketKey is used by TLS servers to provide session
	// resumption. See RFC 5077. If zero, it will be filled with
	// random data before the first server handshake.
	//
	// If multiple servers are terminating connections for the same host
	// they should all have the same SessionTicketKey. If the
	// SessionTicketKey leaks, previously recorded and future TLS
	// connections using that key are compromised.
	SessionTicketKey [32]byte

	// ClientSessionCache is a cache of ClientSessionState entries
	// for TLS session resumption.
	ClientSessionCache ClientSessionCache

	// ServerSessionCache is a cache of sessionState entries for TLS session
	// resumption.
	ServerSessionCache ServerSessionCache

	// MinVersion contains the minimum SSL/TLS version that is acceptable.
	// If zero, then SSLv3 is taken as the minimum.
	MinVersion uint16

	// MaxVersion contains the maximum SSL/TLS version that is acceptable.
	// If zero, then the maximum version supported by this package is used,
	// which is currently TLS 1.2.
	MaxVersion uint16

	// CurvePreferences contains the elliptic curves that will be used in
	// an ECDHE handshake, in preference order. If empty, the default will
	// be used.
	CurvePreferences []CurveID

	// ChannelID contains the ECDSA key for the client to use as
	// its TLS Channel ID.
	ChannelID *ecdsa.PrivateKey

	// RequestChannelID controls whether the server requests a TLS
	// Channel ID. If negotiated, the client's public key is
	// returned in the ConnectionState.
	RequestChannelID bool

	// PreSharedKey, if not nil, is the pre-shared key to use with
	// the PSK cipher suites.
	PreSharedKey []byte

	// PreSharedKeyIdentity, if not empty, is the identity to use
	// with the PSK cipher suites.
	PreSharedKeyIdentity string

	// SRTPProtectionProfiles, if not nil, is the list of SRTP
	// protection profiles to offer in DTLS-SRTP.
	SRTPProtectionProfiles []uint16

	// SignatureAndHashes, if not nil, overrides the default set of
	// supported signature and hash algorithms to advertise in
	// CertificateRequest.
	SignatureAndHashes []signatureAndHash

	// Bugs specifies optional misbehaviour to be used for testing other
	// implementations.
	Bugs ProtocolBugs

	serverInitOnce sync.Once // guards calling (*Config).serverInit
}

type BadValue int

const (
	BadValueNone BadValue = iota
	BadValueNegative
	BadValueZero
	BadValueLimit
	BadValueLarge
	NumBadValues
)

type ProtocolBugs struct {
	// InvalidSKXSignature specifies that the signature in a
	// ServerKeyExchange message should be invalid.
	InvalidSKXSignature bool

	// InvalidCertVerifySignature specifies that the signature in a
	// CertificateVerify message should be invalid.
	InvalidCertVerifySignature bool

	// InvalidSKXCurve causes the curve ID in the ServerKeyExchange message
	// to be wrong.
	InvalidSKXCurve bool

	// BadECDSAR controls ways in which the 'r' value of an ECDSA signature
	// can be invalid.
	BadECDSAR BadValue
	BadECDSAS BadValue

	// MaxPadding causes CBC records to have the maximum possible padding.
	MaxPadding bool
	// PaddingFirstByteBad causes the first byte of the padding to be
	// incorrect.
	PaddingFirstByteBad bool
	// PaddingFirstByteBadIf255 causes the first byte of padding to be
	// incorrect if there's a maximum amount of padding (i.e. 255 bytes).
	PaddingFirstByteBadIf255 bool

	// FailIfNotFallbackSCSV causes a server handshake to fail if the
	// client doesn't send the fallback SCSV value.
	FailIfNotFallbackSCSV bool

	// DuplicateExtension causes an extra empty extension of bogus type to
	// be emitted in either the ClientHello or the ServerHello.
	DuplicateExtension bool

	// UnauthenticatedECDH causes the server to pretend ECDHE_RSA
	// and ECDHE_ECDSA cipher suites are actually ECDH_anon. No
	// Certificate message is sent and no signature is added to
	// ServerKeyExchange.
	UnauthenticatedECDH bool

	// SkipHelloVerifyRequest causes a DTLS server to skip the
	// HelloVerifyRequest message.
	SkipHelloVerifyRequest bool

	// SkipCertificateStatus, if true, causes the server to skip the
	// CertificateStatus message. This is legal because CertificateStatus is
	// optional, even with a status_request in ServerHello.
	SkipCertificateStatus bool

	// SkipServerKeyExchange causes the server to skip sending
	// ServerKeyExchange messages.
	SkipServerKeyExchange bool

	// SkipNewSessionTicket causes the server to skip sending the
	// NewSessionTicket message despite promising to in ServerHello.
	SkipNewSessionTicket bool

	// SkipChangeCipherSpec causes the implementation to skip
	// sending the ChangeCipherSpec message (and adjusting cipher
	// state accordingly for the Finished message).
	SkipChangeCipherSpec bool

	// SkipFinished causes the implementation to skip sending the Finished
	// message.
	SkipFinished bool

	// EarlyChangeCipherSpec causes the client to send an early
	// ChangeCipherSpec message before the ClientKeyExchange. A value of
	// zero disables this behavior. One and two configure variants for 0.9.8
	// and 1.0.1 modes, respectively.
	EarlyChangeCipherSpec int

	// FragmentAcrossChangeCipherSpec causes the implementation to fragment
	// the Finished (or NextProto) message around the ChangeCipherSpec
	// messages.
	FragmentAcrossChangeCipherSpec bool

	// SendV2ClientHello causes the client to send a V2ClientHello
	// instead of a normal ClientHello.
	SendV2ClientHello bool

	// SendFallbackSCSV causes the client to include
	// TLS_FALLBACK_SCSV in the ClientHello.
	SendFallbackSCSV bool

	// SendRenegotiationSCSV causes the client to include the renegotiation
	// SCSV in the ClientHello.
	SendRenegotiationSCSV bool

	// MaxHandshakeRecordLength, if non-zero, is the maximum size of a
	// handshake record. Handshake messages will be split into multiple
	// records at the specified size, except that the client_version will
	// never be fragmented. For DTLS, it is the maximum handshake fragment
	// size, not record size; DTLS allows multiple handshake fragments in a
	// single handshake record. See |PackHandshakeFragments|.
	MaxHandshakeRecordLength int

	// FragmentClientVersion will allow MaxHandshakeRecordLength to apply to
	// the first 6 bytes of the ClientHello.
	FragmentClientVersion bool

	// FragmentAlert will cause all alerts to be fragmented across
	// two records.
	FragmentAlert bool

	// SendSpuriousAlert, if non-zero, will cause an spurious, unwanted
	// alert to be sent.
	SendSpuriousAlert alert

	// RsaClientKeyExchangeVersion, if non-zero, causes the client to send a
	// ClientKeyExchange with the specified version rather than the
	// client_version when performing the RSA key exchange.
	RsaClientKeyExchangeVersion uint16

	// RenewTicketOnResume causes the server to renew the session ticket and
	// send a NewSessionTicket message during an abbreviated handshake.
	RenewTicketOnResume bool

	// SendClientVersion, if non-zero, causes the client to send a different
	// TLS version in the ClientHello than the maximum supported version.
	SendClientVersion uint16

	// ExpectFalseStart causes the server to, on full handshakes,
	// expect the peer to False Start; the server Finished message
	// isn't sent until we receive an application data record
	// from the peer.
	ExpectFalseStart bool

	// AlertBeforeFalseStartTest, if non-zero, causes the server to, on full
	// handshakes, send an alert just before reading the application data
	// record to test False Start. This can be used in a negative False
	// Start test to determine whether the peer processed the alert (and
	// closed the connection) before or after sending app data.
	AlertBeforeFalseStartTest alert

	// SSL3RSAKeyExchange causes the client to always send an RSA
	// ClientKeyExchange message without the two-byte length
	// prefix, as if it were SSL3.
	SSL3RSAKeyExchange bool

	// SkipCipherVersionCheck causes the server to negotiate
	// TLS 1.2 ciphers in earlier versions of TLS.
	SkipCipherVersionCheck bool

	// ExpectServerName, if not empty, is the hostname the client
	// must specify in the server_name extension.
	ExpectServerName string

	// SwapNPNAndALPN switches the relative order between NPN and ALPN in
	// both ClientHello and ServerHello.
	SwapNPNAndALPN bool

	// ALPNProtocol, if not nil, sets the ALPN protocol that a server will
	// return.
	ALPNProtocol *string

	// AllowSessionVersionMismatch causes the server to resume sessions
	// regardless of the version associated with the session.
	AllowSessionVersionMismatch bool

	// CorruptTicket causes a client to corrupt a session ticket before
	// sending it in a resume handshake.
	CorruptTicket bool

	// OversizedSessionId causes the session id that is sent with a ticket
	// resumption attempt to be too large (33 bytes).
	OversizedSessionId bool

	// RequireExtendedMasterSecret, if true, requires that the peer support
	// the extended master secret option.
	RequireExtendedMasterSecret bool

	// NoExtendedMasterSecret causes the client and server to behave as if
	// they didn't support an extended master secret.
	NoExtendedMasterSecret bool

	// EmptyRenegotiationInfo causes the renegotiation extension to be
	// empty in a renegotiation handshake.
	EmptyRenegotiationInfo bool

	// BadRenegotiationInfo causes the renegotiation extension value in a
	// renegotiation handshake to be incorrect.
	BadRenegotiationInfo bool

	// NoRenegotiationInfo causes the client to behave as if it
	// didn't support the renegotiation info extension.
	NoRenegotiationInfo bool

	// RequireRenegotiationInfo, if true, causes the client to return an
	// error if the server doesn't reply with the renegotiation extension.
	RequireRenegotiationInfo bool

	// SequenceNumberMapping, if non-nil, is the mapping function to apply
	// to the sequence number of outgoing packets. For both TLS and DTLS,
	// the two most-significant bytes in the resulting sequence number are
	// ignored so that the DTLS epoch cannot be changed.
	SequenceNumberMapping func(uint64) uint64

	// RSAEphemeralKey, if true, causes the server to send a
	// ServerKeyExchange message containing an ephemeral key (as in
	// RSA_EXPORT) in the plain RSA key exchange.
	RSAEphemeralKey bool

	// SRTPMasterKeyIdentifer, if not empty, is the SRTP MKI value that the
	// client offers when negotiating SRTP. MKI support is still missing so
	// the peer must still send none.
	SRTPMasterKeyIdentifer string

	// SendSRTPProtectionProfile, if non-zero, is the SRTP profile that the
	// server sends in the ServerHello instead of the negotiated one.
	SendSRTPProtectionProfile uint16

	// NoSignatureAndHashes, if true, causes the client to omit the
	// signature and hashes extension.
	//
	// For a server, it will cause an empty list to be sent in the
	// CertificateRequest message. None the less, the configured set will
	// still be enforced.
	NoSignatureAndHashes bool

	// NoSupportedCurves, if true, causes the client to omit the
	// supported_curves extension.
	NoSupportedCurves bool

	// RequireSameRenegoClientVersion, if true, causes the server
	// to require that all ClientHellos match in offered version
	// across a renego.
	RequireSameRenegoClientVersion bool

	// ExpectInitialRecordVersion, if non-zero, is the expected
	// version of the records before the version is determined.
	ExpectInitialRecordVersion uint16

	// MaxPacketLength, if non-zero, is the maximum acceptable size for a
	// packet.
	MaxPacketLength int

	// SendCipherSuite, if non-zero, is the cipher suite value that the
	// server will send in the ServerHello. This does not affect the cipher
	// the server believes it has actually negotiated.
	SendCipherSuite uint16

	// AppDataBeforeHandshake, if not nil, causes application data to be
	// sent immediately before the first handshake message.
	AppDataBeforeHandshake []byte

	// AppDataAfterChangeCipherSpec, if not nil, causes application data to
	// be sent immediately after ChangeCipherSpec.
	AppDataAfterChangeCipherSpec []byte

	// AlertAfterChangeCipherSpec, if non-zero, causes an alert to be sent
	// immediately after ChangeCipherSpec.
	AlertAfterChangeCipherSpec alert

	// TimeoutSchedule is the schedule of packet drops and simulated
	// timeouts for before each handshake leg from the peer.
	TimeoutSchedule []time.Duration

	// PacketAdaptor is the packetAdaptor to use to simulate timeouts.
	PacketAdaptor *packetAdaptor

	// ReorderHandshakeFragments, if true, causes handshake fragments in
	// DTLS to overlap and be sent in the wrong order. It also causes
	// pre-CCS flights to be sent twice. (Post-CCS flights consist of
	// Finished and will trigger a spurious retransmit.)
	ReorderHandshakeFragments bool

	// MixCompleteMessageWithFragments, if true, causes handshake
	// messages in DTLS to redundantly both fragment the message
	// and include a copy of the full one.
	MixCompleteMessageWithFragments bool

	// SendInvalidRecordType, if true, causes a record with an invalid
	// content type to be sent immediately following the handshake.
	SendInvalidRecordType bool

	// WrongCertificateMessageType, if true, causes Certificate message to
	// be sent with the wrong message type.
	WrongCertificateMessageType bool

	// FragmentMessageTypeMismatch, if true, causes all non-initial
	// handshake fragments in DTLS to have the wrong message type.
	FragmentMessageTypeMismatch bool

	// FragmentMessageLengthMismatch, if true, causes all non-initial
	// handshake fragments in DTLS to have the wrong message length.
	FragmentMessageLengthMismatch bool

	// SplitFragments, if non-zero, causes the handshake fragments in DTLS
	// to be split across two records. The value of |SplitFragments| is the
	// number of bytes in the first fragment.
	SplitFragments int

	// SendEmptyFragments, if true, causes handshakes to include empty
	// fragments in DTLS.
	SendEmptyFragments bool

	// SendSplitAlert, if true, causes an alert to be sent with the header
	// and record body split across multiple packets. The peer should
	// discard these packets rather than process it.
	SendSplitAlert bool

	// FailIfResumeOnRenego, if true, causes renegotiations to fail if the
	// client offers a resumption or the server accepts one.
	FailIfResumeOnRenego bool

	// IgnorePeerCipherPreferences, if true, causes the peer's cipher
	// preferences to be ignored.
	IgnorePeerCipherPreferences bool

	// IgnorePeerSignatureAlgorithmPreferences, if true, causes the peer's
	// signature algorithm preferences to be ignored.
	IgnorePeerSignatureAlgorithmPreferences bool

	// IgnorePeerCurvePreferences, if true, causes the peer's curve
	// preferences to be ignored.
	IgnorePeerCurvePreferences bool

	// BadFinished, if true, causes the Finished hash to be broken.
	BadFinished bool

	// DHGroupPrime, if not nil, is used to define the (finite field)
	// Diffie-Hellman group. The generator used is always two.
	DHGroupPrime *big.Int

	// PackHandshakeFragments, if true, causes handshake fragments to be
	// packed into individual handshake records, up to the specified record
	// size.
	PackHandshakeFragments int

	// PackHandshakeRecords, if true, causes handshake records to be packed
	// into individual packets, up to the specified packet size.
	PackHandshakeRecords int

	// EnableAllCiphersInDTLS, if true, causes RC4 to be enabled in DTLS.
	EnableAllCiphersInDTLS bool

	// EmptyCertificateList, if true, causes the server to send an empty
	// certificate list in the Certificate message.
	EmptyCertificateList bool

	// ExpectNewTicket, if true, causes the client to abort if it does not
	// receive a new ticket.
	ExpectNewTicket bool

	// RequireClientHelloSize, if not zero, is the required length in bytes
	// of the ClientHello /record/. This is checked by the server.
	RequireClientHelloSize int

	// CustomExtension, if not empty, contains the contents of an extension
	// that will be added to client/server hellos.
	CustomExtension string

	// ExpectedCustomExtension, if not nil, contains the expected contents
	// of a custom extension.
	ExpectedCustomExtension *string

	// NoCloseNotify, if true, causes the close_notify alert to be skipped
	// on connection shutdown.
	NoCloseNotify bool

	// ExpectCloseNotify, if true, requires a close_notify from the peer on
	// shutdown. Records from the peer received after close_notify is sent
	// are not discard.
	ExpectCloseNotify bool

	// SendLargeRecords, if true, allows outgoing records to be sent
	// arbitrarily large.
	SendLargeRecords bool

	// NegotiateALPNAndNPN, if true, causes the server to negotiate both
	// ALPN and NPN in the same connetion.
	NegotiateALPNAndNPN bool
}

func (c *Config) serverInit() {
	if c.SessionTicketsDisabled {
		return
	}

	// If the key has already been set then we have nothing to do.
	for _, b := range c.SessionTicketKey {
		if b != 0 {
			return
		}
	}

	if _, err := io.ReadFull(c.rand(), c.SessionTicketKey[:]); err != nil {
		c.SessionTicketsDisabled = true
	}
}

func (c *Config) rand() io.Reader {
	r := c.Rand
	if r == nil {
		return rand.Reader
	}
	return r
}

func (c *Config) time() time.Time {
	t := c.Time
	if t == nil {
		t = time.Now
	}
	return t()
}

func (c *Config) cipherSuites() []uint16 {
	s := c.CipherSuites
	if s == nil {
		s = defaultCipherSuites()
	}
	return s
}

func (c *Config) minVersion() uint16 {
	if c == nil || c.MinVersion == 0 {
		return minVersion
	}
	return c.MinVersion
}

func (c *Config) maxVersion() uint16 {
	if c == nil || c.MaxVersion == 0 {
		return maxVersion
	}
	return c.MaxVersion
}

var defaultCurvePreferences = []CurveID{CurveP256, CurveP384, CurveP521}

func (c *Config) curvePreferences() []CurveID {
	if c == nil || len(c.CurvePreferences) == 0 {
		return defaultCurvePreferences
	}
	return c.CurvePreferences
}

// mutualVersion returns the protocol version to use given the advertised
// version of the peer.
func (c *Config) mutualVersion(vers uint16) (uint16, bool) {
	minVersion := c.minVersion()
	maxVersion := c.maxVersion()

	if vers < minVersion {
		return 0, false
	}
	if vers > maxVersion {
		vers = maxVersion
	}
	return vers, true
}

// getCertificateForName returns the best certificate for the given name,
// defaulting to the first element of c.Certificates if there are no good
// options.
func (c *Config) getCertificateForName(name string) *Certificate {
	if len(c.Certificates) == 1 || c.NameToCertificate == nil {
		// There's only one choice, so no point doing any work.
		return &c.Certificates[0]
	}

	name = strings.ToLower(name)
	for len(name) > 0 && name[len(name)-1] == '.' {
		name = name[:len(name)-1]
	}

	if cert, ok := c.NameToCertificate[name]; ok {
		return cert
	}

	// try replacing labels in the name with wildcards until we get a
	// match.
	labels := strings.Split(name, ".")
	for i := range labels {
		labels[i] = "*"
		candidate := strings.Join(labels, ".")
		if cert, ok := c.NameToCertificate[candidate]; ok {
			return cert
		}
	}

	// If nothing matches, return the first certificate.
	return &c.Certificates[0]
}

func (c *Config) signatureAndHashesForServer() []signatureAndHash {
	if c != nil && c.SignatureAndHashes != nil {
		return c.SignatureAndHashes
	}
	return supportedClientCertSignatureAlgorithms
}

func (c *Config) signatureAndHashesForClient() []signatureAndHash {
	if c != nil && c.SignatureAndHashes != nil {
		return c.SignatureAndHashes
	}
	return supportedSKXSignatureAlgorithms
}

// BuildNameToCertificate parses c.Certificates and builds c.NameToCertificate
// from the CommonName and SubjectAlternateName fields of each of the leaf
// certificates.
func (c *Config) BuildNameToCertificate() {
	c.NameToCertificate = make(map[string]*Certificate)
	for i := range c.Certificates {
		cert := &c.Certificates[i]
		x509Cert, err := x509.ParseCertificate(cert.Certificate[0])
		if err != nil {
			continue
		}
		if len(x509Cert.Subject.CommonName) > 0 {
			c.NameToCertificate[x509Cert.Subject.CommonName] = cert
		}
		for _, san := range x509Cert.DNSNames {
			c.NameToCertificate[san] = cert
		}
	}
}

// A Certificate is a chain of one or more certificates, leaf first.
type Certificate struct {
	Certificate [][]byte
	PrivateKey  crypto.PrivateKey // supported types: *rsa.PrivateKey, *ecdsa.PrivateKey
	// OCSPStaple contains an optional OCSP response which will be served
	// to clients that request it.
	OCSPStaple []byte
	// SignedCertificateTimestampList contains an optional encoded
	// SignedCertificateTimestampList structure which will be
	// served to clients that request it.
	SignedCertificateTimestampList []byte
	// Leaf is the parsed form of the leaf certificate, which may be
	// initialized using x509.ParseCertificate to reduce per-handshake
	// processing for TLS clients doing client authentication. If nil, the
	// leaf certificate will be parsed as needed.
	Leaf *x509.Certificate
}

// A TLS record.
type record struct {
	contentType  recordType
	major, minor uint8
	payload      []byte
}

type handshakeMessage interface {
	marshal() []byte
	unmarshal([]byte) bool
}

// lruSessionCache is a client or server session cache implementation
// that uses an LRU caching strategy.
type lruSessionCache struct {
	sync.Mutex

	m        map[string]*list.Element
	q        *list.List
	capacity int
}

type lruSessionCacheEntry struct {
	sessionKey string
	state      interface{}
}

// Put adds the provided (sessionKey, cs) pair to the cache.
func (c *lruSessionCache) Put(sessionKey string, cs interface{}) {
	c.Lock()
	defer c.Unlock()

	if elem, ok := c.m[sessionKey]; ok {
		entry := elem.Value.(*lruSessionCacheEntry)
		entry.state = cs
		c.q.MoveToFront(elem)
		return
	}

	if c.q.Len() < c.capacity {
		entry := &lruSessionCacheEntry{sessionKey, cs}
		c.m[sessionKey] = c.q.PushFront(entry)
		return
	}

	elem := c.q.Back()
	entry := elem.Value.(*lruSessionCacheEntry)
	delete(c.m, entry.sessionKey)
	entry.sessionKey = sessionKey
	entry.state = cs
	c.q.MoveToFront(elem)
	c.m[sessionKey] = elem
}

// Get returns the value associated with a given key. It returns (nil,
// false) if no value is found.
func (c *lruSessionCache) Get(sessionKey string) (interface{}, bool) {
	c.Lock()
	defer c.Unlock()

	if elem, ok := c.m[sessionKey]; ok {
		c.q.MoveToFront(elem)
		return elem.Value.(*lruSessionCacheEntry).state, true
	}
	return nil, false
}

// lruClientSessionCache is a ClientSessionCache implementation that
// uses an LRU caching strategy.
type lruClientSessionCache struct {
	lruSessionCache
}

func (c *lruClientSessionCache) Put(sessionKey string, cs *ClientSessionState) {
	c.lruSessionCache.Put(sessionKey, cs)
}

func (c *lruClientSessionCache) Get(sessionKey string) (*ClientSessionState, bool) {
	cs, ok := c.lruSessionCache.Get(sessionKey)
	if !ok {
		return nil, false
	}
	return cs.(*ClientSessionState), true
}

// lruServerSessionCache is a ServerSessionCache implementation that
// uses an LRU caching strategy.
type lruServerSessionCache struct {
	lruSessionCache
}

func (c *lruServerSessionCache) Put(sessionId string, session *sessionState) {
	c.lruSessionCache.Put(sessionId, session)
}

func (c *lruServerSessionCache) Get(sessionId string) (*sessionState, bool) {
	cs, ok := c.lruSessionCache.Get(sessionId)
	if !ok {
		return nil, false
	}
	return cs.(*sessionState), true
}

// NewLRUClientSessionCache returns a ClientSessionCache with the given
// capacity that uses an LRU strategy. If capacity is < 1, a default capacity
// is used instead.
func NewLRUClientSessionCache(capacity int) ClientSessionCache {
	const defaultSessionCacheCapacity = 64

	if capacity < 1 {
		capacity = defaultSessionCacheCapacity
	}
	return &lruClientSessionCache{
		lruSessionCache{
			m:        make(map[string]*list.Element),
			q:        list.New(),
			capacity: capacity,
		},
	}
}

// NewLRUServerSessionCache returns a ServerSessionCache with the given
// capacity that uses an LRU strategy. If capacity is < 1, a default capacity
// is used instead.
func NewLRUServerSessionCache(capacity int) ServerSessionCache {
	const defaultSessionCacheCapacity = 64

	if capacity < 1 {
		capacity = defaultSessionCacheCapacity
	}
	return &lruServerSessionCache{
		lruSessionCache{
			m:        make(map[string]*list.Element),
			q:        list.New(),
			capacity: capacity,
		},
	}
}

// TODO(jsing): Make these available to both crypto/x509 and crypto/tls.
type dsaSignature struct {
	R, S *big.Int
}

type ecdsaSignature dsaSignature

var emptyConfig Config

func defaultConfig() *Config {
	return &emptyConfig
}

var (
	once                   sync.Once
	varDefaultCipherSuites []uint16
)

func defaultCipherSuites() []uint16 {
	once.Do(initDefaultCipherSuites)
	return varDefaultCipherSuites
}

func initDefaultCipherSuites() {
	for _, suite := range cipherSuites {
		if suite.flags&suitePSK == 0 {
			varDefaultCipherSuites = append(varDefaultCipherSuites, suite.id)
		}
	}
}

func unexpectedMessageError(wanted, got interface{}) error {
	return fmt.Errorf("tls: received unexpected handshake message of type %T when waiting for %T", got, wanted)
}

func isSupportedSignatureAndHash(sigHash signatureAndHash, sigHashes []signatureAndHash) bool {
	for _, s := range sigHashes {
		if s == sigHash {
			return true
		}
	}
	return false
}
