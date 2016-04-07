// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runner

import (
	"bytes"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rsa"
	"crypto/subtle"
	"crypto/x509"
	"encoding/asn1"
	"errors"
	"fmt"
	"io"
	"math/big"
	"net"
	"strconv"
)

type clientHandshakeState struct {
	c             *Conn
	serverHello   *serverHelloMsg
	hello         *clientHelloMsg
	suite         *cipherSuite
	finishedHash  finishedHash
	masterSecret  []byte
	session       *ClientSessionState
	finishedBytes []byte
}

func (c *Conn) clientHandshake() error {
	if c.config == nil {
		c.config = defaultConfig()
	}

	if len(c.config.ServerName) == 0 && !c.config.InsecureSkipVerify {
		return errors.New("tls: either ServerName or InsecureSkipVerify must be specified in the tls.Config")
	}

	c.sendHandshakeSeq = 0
	c.recvHandshakeSeq = 0

	nextProtosLength := 0
	for _, proto := range c.config.NextProtos {
		if l := len(proto); l > 255 {
			return errors.New("tls: invalid NextProtos value")
		} else {
			nextProtosLength += 1 + l
		}
	}
	if nextProtosLength > 0xffff {
		return errors.New("tls: NextProtos values too large")
	}

	hello := &clientHelloMsg{
		isDTLS:                  c.isDTLS,
		vers:                    c.config.maxVersion(),
		compressionMethods:      []uint8{compressionNone},
		random:                  make([]byte, 32),
		ocspStapling:            true,
		sctListSupported:        true,
		serverName:              c.config.ServerName,
		supportedCurves:         c.config.curvePreferences(),
		supportedPoints:         []uint8{pointFormatUncompressed},
		nextProtoNeg:            len(c.config.NextProtos) > 0,
		secureRenegotiation:     []byte{},
		alpnProtocols:           c.config.NextProtos,
		duplicateExtension:      c.config.Bugs.DuplicateExtension,
		channelIDSupported:      c.config.ChannelID != nil,
		npnLast:                 c.config.Bugs.SwapNPNAndALPN,
		extendedMasterSecret:    c.config.maxVersion() >= VersionTLS10,
		srtpProtectionProfiles:  c.config.SRTPProtectionProfiles,
		srtpMasterKeyIdentifier: c.config.Bugs.SRTPMasterKeyIdentifer,
		customExtension:         c.config.Bugs.CustomExtension,
	}

	if c.config.Bugs.SendClientVersion != 0 {
		hello.vers = c.config.Bugs.SendClientVersion
	}

	if c.config.Bugs.NoExtendedMasterSecret {
		hello.extendedMasterSecret = false
	}

	if c.config.Bugs.NoSupportedCurves {
		hello.supportedCurves = nil
	}

	if len(c.clientVerify) > 0 && !c.config.Bugs.EmptyRenegotiationInfo {
		if c.config.Bugs.BadRenegotiationInfo {
			hello.secureRenegotiation = append(hello.secureRenegotiation, c.clientVerify...)
			hello.secureRenegotiation[0] ^= 0x80
		} else {
			hello.secureRenegotiation = c.clientVerify
		}
	}

	if c.config.Bugs.NoRenegotiationInfo {
		hello.secureRenegotiation = nil
	}

	possibleCipherSuites := c.config.cipherSuites()
	hello.cipherSuites = make([]uint16, 0, len(possibleCipherSuites))

NextCipherSuite:
	for _, suiteId := range possibleCipherSuites {
		for _, suite := range cipherSuites {
			if suite.id != suiteId {
				continue
			}
			// Don't advertise TLS 1.2-only cipher suites unless
			// we're attempting TLS 1.2.
			if hello.vers < VersionTLS12 && suite.flags&suiteTLS12 != 0 {
				continue
			}
			// Don't advertise non-DTLS cipher suites on DTLS.
			if c.isDTLS && suite.flags&suiteNoDTLS != 0 && !c.config.Bugs.EnableAllCiphersInDTLS {
				continue
			}
			hello.cipherSuites = append(hello.cipherSuites, suiteId)
			continue NextCipherSuite
		}
	}

	if c.config.Bugs.SendRenegotiationSCSV {
		hello.cipherSuites = append(hello.cipherSuites, renegotiationSCSV)
	}

	if c.config.Bugs.SendFallbackSCSV {
		hello.cipherSuites = append(hello.cipherSuites, fallbackSCSV)
	}

	_, err := io.ReadFull(c.config.rand(), hello.random)
	if err != nil {
		c.sendAlert(alertInternalError)
		return errors.New("tls: short read from Rand: " + err.Error())
	}

	if hello.vers >= VersionTLS12 && !c.config.Bugs.NoSignatureAndHashes {
		hello.signatureAndHashes = c.config.signatureAndHashesForClient()
	}

	var session *ClientSessionState
	var cacheKey string
	sessionCache := c.config.ClientSessionCache

	if sessionCache != nil {
		hello.ticketSupported = !c.config.SessionTicketsDisabled

		// Try to resume a previously negotiated TLS session, if
		// available.
		cacheKey = clientSessionCacheKey(c.conn.RemoteAddr(), c.config)
		candidateSession, ok := sessionCache.Get(cacheKey)
		if ok {
			ticketOk := !c.config.SessionTicketsDisabled || candidateSession.sessionTicket == nil

			// Check that the ciphersuite/version used for the
			// previous session are still valid.
			cipherSuiteOk := false
			for _, id := range hello.cipherSuites {
				if id == candidateSession.cipherSuite {
					cipherSuiteOk = true
					break
				}
			}

			versOk := candidateSession.vers >= c.config.minVersion() &&
				candidateSession.vers <= c.config.maxVersion()
			if ticketOk && versOk && cipherSuiteOk {
				session = candidateSession
			}
		}
	}

	if session != nil {
		if session.sessionTicket != nil {
			hello.sessionTicket = session.sessionTicket
			if c.config.Bugs.CorruptTicket {
				hello.sessionTicket = make([]byte, len(session.sessionTicket))
				copy(hello.sessionTicket, session.sessionTicket)
				if len(hello.sessionTicket) > 0 {
					offset := 40
					if offset > len(hello.sessionTicket) {
						offset = len(hello.sessionTicket) - 1
					}
					hello.sessionTicket[offset] ^= 0x40
				}
			}
			// A random session ID is used to detect when the
			// server accepted the ticket and is resuming a session
			// (see RFC 5077).
			sessionIdLen := 16
			if c.config.Bugs.OversizedSessionId {
				sessionIdLen = 33
			}
			hello.sessionId = make([]byte, sessionIdLen)
			if _, err := io.ReadFull(c.config.rand(), hello.sessionId); err != nil {
				c.sendAlert(alertInternalError)
				return errors.New("tls: short read from Rand: " + err.Error())
			}
		} else {
			hello.sessionId = session.sessionId
		}
	}

	var helloBytes []byte
	if c.config.Bugs.SendV2ClientHello {
		// Test that the peer left-pads random.
		hello.random[0] = 0
		v2Hello := &v2ClientHelloMsg{
			vers:         hello.vers,
			cipherSuites: hello.cipherSuites,
			// No session resumption for V2ClientHello.
			sessionId: nil,
			challenge: hello.random[1:],
		}
		helloBytes = v2Hello.marshal()
		c.writeV2Record(helloBytes)
	} else {
		helloBytes = hello.marshal()
		c.writeRecord(recordTypeHandshake, helloBytes)
	}
	c.dtlsFlushHandshake()

	if err := c.simulatePacketLoss(nil); err != nil {
		return err
	}
	msg, err := c.readHandshake()
	if err != nil {
		return err
	}

	if c.isDTLS {
		helloVerifyRequest, ok := msg.(*helloVerifyRequestMsg)
		if ok {
			if helloVerifyRequest.vers != VersionTLS10 {
				// Per RFC 6347, the version field in
				// HelloVerifyRequest SHOULD be always DTLS
				// 1.0. Enforce this for testing purposes.
				return errors.New("dtls: bad HelloVerifyRequest version")
			}

			hello.raw = nil
			hello.cookie = helloVerifyRequest.cookie
			helloBytes = hello.marshal()
			c.writeRecord(recordTypeHandshake, helloBytes)
			c.dtlsFlushHandshake()

			if err := c.simulatePacketLoss(nil); err != nil {
				return err
			}
			msg, err = c.readHandshake()
			if err != nil {
				return err
			}
		}
	}

	serverHello, ok := msg.(*serverHelloMsg)
	if !ok {
		c.sendAlert(alertUnexpectedMessage)
		return unexpectedMessageError(serverHello, msg)
	}

	c.vers, ok = c.config.mutualVersion(serverHello.vers)
	if !ok {
		c.sendAlert(alertProtocolVersion)
		return fmt.Errorf("tls: server selected unsupported protocol version %x", serverHello.vers)
	}
	c.haveVers = true

	suite := mutualCipherSuite(c.config.cipherSuites(), serverHello.cipherSuite)
	if suite == nil {
		c.sendAlert(alertHandshakeFailure)
		return fmt.Errorf("tls: server selected an unsupported cipher suite")
	}

	if c.config.Bugs.RequireRenegotiationInfo && serverHello.secureRenegotiation == nil {
		return errors.New("tls: renegotiation extension missing")
	}

	if len(c.clientVerify) > 0 && !c.config.Bugs.NoRenegotiationInfo {
		var expectedRenegInfo []byte
		expectedRenegInfo = append(expectedRenegInfo, c.clientVerify...)
		expectedRenegInfo = append(expectedRenegInfo, c.serverVerify...)
		if !bytes.Equal(serverHello.secureRenegotiation, expectedRenegInfo) {
			c.sendAlert(alertHandshakeFailure)
			return fmt.Errorf("tls: renegotiation mismatch")
		}
	}

	if expected := c.config.Bugs.ExpectedCustomExtension; expected != nil {
		if serverHello.customExtension != *expected {
			return fmt.Errorf("tls: bad custom extension contents %q", serverHello.customExtension)
		}
	}

	hs := &clientHandshakeState{
		c:            c,
		serverHello:  serverHello,
		hello:        hello,
		suite:        suite,
		finishedHash: newFinishedHash(c.vers, suite),
		session:      session,
	}

	hs.writeHash(helloBytes, hs.c.sendHandshakeSeq-1)
	hs.writeServerHash(hs.serverHello.marshal())

	if c.config.Bugs.EarlyChangeCipherSpec > 0 {
		hs.establishKeys()
		c.writeRecord(recordTypeChangeCipherSpec, []byte{1})
	}

	isResume, err := hs.processServerHello()
	if err != nil {
		return err
	}

	if isResume {
		if c.config.Bugs.EarlyChangeCipherSpec == 0 {
			if err := hs.establishKeys(); err != nil {
				return err
			}
		}
		if err := hs.readSessionTicket(); err != nil {
			return err
		}
		if err := hs.readFinished(c.firstFinished[:]); err != nil {
			return err
		}
		if err := hs.sendFinished(nil, isResume); err != nil {
			return err
		}
	} else {
		if err := hs.doFullHandshake(); err != nil {
			return err
		}
		if err := hs.establishKeys(); err != nil {
			return err
		}
		if err := hs.sendFinished(c.firstFinished[:], isResume); err != nil {
			return err
		}
		// Most retransmits are triggered by a timeout, but the final
		// leg of the handshake is retransmited upon re-receiving a
		// Finished.
		if err := c.simulatePacketLoss(func() {
			c.writeRecord(recordTypeHandshake, hs.finishedBytes)
			c.dtlsFlushHandshake()
		}); err != nil {
			return err
		}
		if err := hs.readSessionTicket(); err != nil {
			return err
		}
		if err := hs.readFinished(nil); err != nil {
			return err
		}
	}

	if sessionCache != nil && hs.session != nil && session != hs.session {
		sessionCache.Put(cacheKey, hs.session)
	}

	c.didResume = isResume
	c.handshakeComplete = true
	c.cipherSuite = suite
	copy(c.clientRandom[:], hs.hello.random)
	copy(c.serverRandom[:], hs.serverHello.random)
	copy(c.masterSecret[:], hs.masterSecret)

	return nil
}

func (hs *clientHandshakeState) doFullHandshake() error {
	c := hs.c

	var leaf *x509.Certificate
	if hs.suite.flags&suitePSK == 0 {
		msg, err := c.readHandshake()
		if err != nil {
			return err
		}

		certMsg, ok := msg.(*certificateMsg)
		if !ok || len(certMsg.certificates) == 0 {
			c.sendAlert(alertUnexpectedMessage)
			return unexpectedMessageError(certMsg, msg)
		}
		hs.writeServerHash(certMsg.marshal())

		certs := make([]*x509.Certificate, len(certMsg.certificates))
		for i, asn1Data := range certMsg.certificates {
			cert, err := x509.ParseCertificate(asn1Data)
			if err != nil {
				c.sendAlert(alertBadCertificate)
				return errors.New("tls: failed to parse certificate from server: " + err.Error())
			}
			certs[i] = cert
		}
		leaf = certs[0]

		if !c.config.InsecureSkipVerify {
			opts := x509.VerifyOptions{
				Roots:         c.config.RootCAs,
				CurrentTime:   c.config.time(),
				DNSName:       c.config.ServerName,
				Intermediates: x509.NewCertPool(),
			}

			for i, cert := range certs {
				if i == 0 {
					continue
				}
				opts.Intermediates.AddCert(cert)
			}
			c.verifiedChains, err = leaf.Verify(opts)
			if err != nil {
				c.sendAlert(alertBadCertificate)
				return err
			}
		}

		switch leaf.PublicKey.(type) {
		case *rsa.PublicKey, *ecdsa.PublicKey:
			break
		default:
			c.sendAlert(alertUnsupportedCertificate)
			return fmt.Errorf("tls: server's certificate contains an unsupported type of public key: %T", leaf.PublicKey)
		}

		c.peerCertificates = certs
	}

	if hs.serverHello.ocspStapling {
		msg, err := c.readHandshake()
		if err != nil {
			return err
		}
		cs, ok := msg.(*certificateStatusMsg)
		if !ok {
			c.sendAlert(alertUnexpectedMessage)
			return unexpectedMessageError(cs, msg)
		}
		hs.writeServerHash(cs.marshal())

		if cs.statusType == statusTypeOCSP {
			c.ocspResponse = cs.response
		}
	}

	msg, err := c.readHandshake()
	if err != nil {
		return err
	}

	keyAgreement := hs.suite.ka(c.vers)

	skx, ok := msg.(*serverKeyExchangeMsg)
	if ok {
		hs.writeServerHash(skx.marshal())
		err = keyAgreement.processServerKeyExchange(c.config, hs.hello, hs.serverHello, leaf, skx)
		if err != nil {
			c.sendAlert(alertUnexpectedMessage)
			return err
		}

		msg, err = c.readHandshake()
		if err != nil {
			return err
		}
	}

	var chainToSend *Certificate
	var certRequested bool
	certReq, ok := msg.(*certificateRequestMsg)
	if ok {
		certRequested = true

		// RFC 4346 on the certificateAuthorities field:
		// A list of the distinguished names of acceptable certificate
		// authorities. These distinguished names may specify a desired
		// distinguished name for a root CA or for a subordinate CA;
		// thus, this message can be used to describe both known roots
		// and a desired authorization space. If the
		// certificate_authorities list is empty then the client MAY
		// send any certificate of the appropriate
		// ClientCertificateType, unless there is some external
		// arrangement to the contrary.

		hs.writeServerHash(certReq.marshal())

		var rsaAvail, ecdsaAvail bool
		for _, certType := range certReq.certificateTypes {
			switch certType {
			case CertTypeRSASign:
				rsaAvail = true
			case CertTypeECDSASign:
				ecdsaAvail = true
			}
		}

		// We need to search our list of client certs for one
		// where SignatureAlgorithm is RSA and the Issuer is in
		// certReq.certificateAuthorities
	findCert:
		for i, chain := range c.config.Certificates {
			if !rsaAvail && !ecdsaAvail {
				continue
			}

			for j, cert := range chain.Certificate {
				x509Cert := chain.Leaf
				// parse the certificate if this isn't the leaf
				// node, or if chain.Leaf was nil
				if j != 0 || x509Cert == nil {
					if x509Cert, err = x509.ParseCertificate(cert); err != nil {
						c.sendAlert(alertInternalError)
						return errors.New("tls: failed to parse client certificate #" + strconv.Itoa(i) + ": " + err.Error())
					}
				}

				switch {
				case rsaAvail && x509Cert.PublicKeyAlgorithm == x509.RSA:
				case ecdsaAvail && x509Cert.PublicKeyAlgorithm == x509.ECDSA:
				default:
					continue findCert
				}

				if len(certReq.certificateAuthorities) == 0 {
					// they gave us an empty list, so just take the
					// first RSA cert from c.config.Certificates
					chainToSend = &chain
					break findCert
				}

				for _, ca := range certReq.certificateAuthorities {
					if bytes.Equal(x509Cert.RawIssuer, ca) {
						chainToSend = &chain
						break findCert
					}
				}
			}
		}

		msg, err = c.readHandshake()
		if err != nil {
			return err
		}
	}

	shd, ok := msg.(*serverHelloDoneMsg)
	if !ok {
		c.sendAlert(alertUnexpectedMessage)
		return unexpectedMessageError(shd, msg)
	}
	hs.writeServerHash(shd.marshal())

	// If the server requested a certificate then we have to send a
	// Certificate message, even if it's empty because we don't have a
	// certificate to send.
	if certRequested {
		certMsg := new(certificateMsg)
		if chainToSend != nil {
			certMsg.certificates = chainToSend.Certificate
		}
		hs.writeClientHash(certMsg.marshal())
		c.writeRecord(recordTypeHandshake, certMsg.marshal())
	}

	preMasterSecret, ckx, err := keyAgreement.generateClientKeyExchange(c.config, hs.hello, leaf)
	if err != nil {
		c.sendAlert(alertInternalError)
		return err
	}
	if ckx != nil {
		if c.config.Bugs.EarlyChangeCipherSpec < 2 {
			hs.writeClientHash(ckx.marshal())
		}
		c.writeRecord(recordTypeHandshake, ckx.marshal())
	}

	if hs.serverHello.extendedMasterSecret && c.vers >= VersionTLS10 {
		hs.masterSecret = extendedMasterFromPreMasterSecret(c.vers, hs.suite, preMasterSecret, hs.finishedHash)
		c.extendedMasterSecret = true
	} else {
		if c.config.Bugs.RequireExtendedMasterSecret {
			return errors.New("tls: extended master secret required but not supported by peer")
		}
		hs.masterSecret = masterFromPreMasterSecret(c.vers, hs.suite, preMasterSecret, hs.hello.random, hs.serverHello.random)
	}

	if chainToSend != nil {
		var signed []byte
		certVerify := &certificateVerifyMsg{
			hasSignatureAndHash: c.vers >= VersionTLS12,
		}

		// Determine the hash to sign.
		var signatureType uint8
		switch c.config.Certificates[0].PrivateKey.(type) {
		case *ecdsa.PrivateKey:
			signatureType = signatureECDSA
		case *rsa.PrivateKey:
			signatureType = signatureRSA
		default:
			c.sendAlert(alertInternalError)
			return errors.New("unknown private key type")
		}
		if c.config.Bugs.IgnorePeerSignatureAlgorithmPreferences {
			certReq.signatureAndHashes = c.config.signatureAndHashesForClient()
		}
		certVerify.signatureAndHash, err = hs.finishedHash.selectClientCertSignatureAlgorithm(certReq.signatureAndHashes, c.config.signatureAndHashesForClient(), signatureType)
		if err != nil {
			c.sendAlert(alertInternalError)
			return err
		}
		digest, hashFunc, err := hs.finishedHash.hashForClientCertificate(certVerify.signatureAndHash, hs.masterSecret)
		if err != nil {
			c.sendAlert(alertInternalError)
			return err
		}
		if c.config.Bugs.InvalidCertVerifySignature {
			digest[0] ^= 0x80
		}

		switch key := c.config.Certificates[0].PrivateKey.(type) {
		case *ecdsa.PrivateKey:
			var r, s *big.Int
			r, s, err = ecdsa.Sign(c.config.rand(), key, digest)
			if err == nil {
				signed, err = asn1.Marshal(ecdsaSignature{r, s})
			}
		case *rsa.PrivateKey:
			signed, err = rsa.SignPKCS1v15(c.config.rand(), key, hashFunc, digest)
		default:
			err = errors.New("unknown private key type")
		}
		if err != nil {
			c.sendAlert(alertInternalError)
			return errors.New("tls: failed to sign handshake with client certificate: " + err.Error())
		}
		certVerify.signature = signed

		hs.writeClientHash(certVerify.marshal())
		c.writeRecord(recordTypeHandshake, certVerify.marshal())
	}
	c.dtlsFlushHandshake()

	hs.finishedHash.discardHandshakeBuffer()

	return nil
}

func (hs *clientHandshakeState) establishKeys() error {
	c := hs.c

	clientMAC, serverMAC, clientKey, serverKey, clientIV, serverIV :=
		keysFromMasterSecret(c.vers, hs.suite, hs.masterSecret, hs.hello.random, hs.serverHello.random, hs.suite.macLen, hs.suite.keyLen, hs.suite.ivLen)
	var clientCipher, serverCipher interface{}
	var clientHash, serverHash macFunction
	if hs.suite.cipher != nil {
		clientCipher = hs.suite.cipher(clientKey, clientIV, false /* not for reading */)
		clientHash = hs.suite.mac(c.vers, clientMAC)
		serverCipher = hs.suite.cipher(serverKey, serverIV, true /* for reading */)
		serverHash = hs.suite.mac(c.vers, serverMAC)
	} else {
		clientCipher = hs.suite.aead(clientKey, clientIV)
		serverCipher = hs.suite.aead(serverKey, serverIV)
	}

	c.in.prepareCipherSpec(c.vers, serverCipher, serverHash)
	c.out.prepareCipherSpec(c.vers, clientCipher, clientHash)
	return nil
}

func (hs *clientHandshakeState) serverResumedSession() bool {
	// If the server responded with the same sessionId then it means the
	// sessionTicket is being used to resume a TLS session.
	return hs.session != nil && hs.hello.sessionId != nil &&
		bytes.Equal(hs.serverHello.sessionId, hs.hello.sessionId)
}

func (hs *clientHandshakeState) processServerHello() (bool, error) {
	c := hs.c

	if hs.serverHello.compressionMethod != compressionNone {
		c.sendAlert(alertUnexpectedMessage)
		return false, errors.New("tls: server selected unsupported compression format")
	}

	clientDidNPN := hs.hello.nextProtoNeg
	clientDidALPN := len(hs.hello.alpnProtocols) > 0
	serverHasNPN := hs.serverHello.nextProtoNeg
	serverHasALPN := len(hs.serverHello.alpnProtocol) > 0

	if !clientDidNPN && serverHasNPN {
		c.sendAlert(alertHandshakeFailure)
		return false, errors.New("server advertised unrequested NPN extension")
	}

	if !clientDidALPN && serverHasALPN {
		c.sendAlert(alertHandshakeFailure)
		return false, errors.New("server advertised unrequested ALPN extension")
	}

	if serverHasNPN && serverHasALPN {
		c.sendAlert(alertHandshakeFailure)
		return false, errors.New("server advertised both NPN and ALPN extensions")
	}

	if serverHasALPN {
		c.clientProtocol = hs.serverHello.alpnProtocol
		c.clientProtocolFallback = false
		c.usedALPN = true
	}

	if !hs.hello.channelIDSupported && hs.serverHello.channelIDRequested {
		c.sendAlert(alertHandshakeFailure)
		return false, errors.New("server advertised unrequested Channel ID extension")
	}

	if hs.serverHello.srtpProtectionProfile != 0 {
		if hs.serverHello.srtpMasterKeyIdentifier != "" {
			return false, errors.New("tls: server selected SRTP MKI value")
		}

		found := false
		for _, p := range c.config.SRTPProtectionProfiles {
			if p == hs.serverHello.srtpProtectionProfile {
				found = true
				break
			}
		}
		if !found {
			return false, errors.New("tls: server advertised unsupported SRTP profile")
		}

		c.srtpProtectionProfile = hs.serverHello.srtpProtectionProfile
	}

	if hs.serverResumedSession() {
		// For test purposes, assert that the server never accepts the
		// resumption offer on renegotiation.
		if c.cipherSuite != nil && c.config.Bugs.FailIfResumeOnRenego {
			return false, errors.New("tls: server resumed session on renegotiation")
		}

		if hs.serverHello.sctList != nil {
			return false, errors.New("tls: server sent SCT extension on session resumption")
		}

		if hs.serverHello.ocspStapling {
			return false, errors.New("tls: server sent OCSP extension on session resumption")
		}

		// Restore masterSecret and peerCerts from previous state
		hs.masterSecret = hs.session.masterSecret
		c.peerCertificates = hs.session.serverCertificates
		c.extendedMasterSecret = hs.session.extendedMasterSecret
		c.sctList = hs.session.sctList
		c.ocspResponse = hs.session.ocspResponse
		hs.finishedHash.discardHandshakeBuffer()
		return true, nil
	}

	if hs.serverHello.sctList != nil {
		c.sctList = hs.serverHello.sctList
	}

	return false, nil
}

func (hs *clientHandshakeState) readFinished(out []byte) error {
	c := hs.c

	c.readRecord(recordTypeChangeCipherSpec)
	if err := c.in.error(); err != nil {
		return err
	}

	msg, err := c.readHandshake()
	if err != nil {
		return err
	}
	serverFinished, ok := msg.(*finishedMsg)
	if !ok {
		c.sendAlert(alertUnexpectedMessage)
		return unexpectedMessageError(serverFinished, msg)
	}

	if c.config.Bugs.EarlyChangeCipherSpec == 0 {
		verify := hs.finishedHash.serverSum(hs.masterSecret)
		if len(verify) != len(serverFinished.verifyData) ||
			subtle.ConstantTimeCompare(verify, serverFinished.verifyData) != 1 {
			c.sendAlert(alertHandshakeFailure)
			return errors.New("tls: server's Finished message was incorrect")
		}
	}
	c.serverVerify = append(c.serverVerify[:0], serverFinished.verifyData...)
	copy(out, serverFinished.verifyData)
	hs.writeServerHash(serverFinished.marshal())
	return nil
}

func (hs *clientHandshakeState) readSessionTicket() error {
	c := hs.c

	// Create a session with no server identifier. Either a
	// session ID or session ticket will be attached.
	session := &ClientSessionState{
		vers:               c.vers,
		cipherSuite:        hs.suite.id,
		masterSecret:       hs.masterSecret,
		handshakeHash:      hs.finishedHash.server.Sum(nil),
		serverCertificates: c.peerCertificates,
		sctList:            c.sctList,
		ocspResponse:       c.ocspResponse,
	}

	if !hs.serverHello.ticketSupported {
		if c.config.Bugs.ExpectNewTicket {
			return errors.New("tls: expected new ticket")
		}
		if hs.session == nil && len(hs.serverHello.sessionId) > 0 {
			session.sessionId = hs.serverHello.sessionId
			hs.session = session
		}
		return nil
	}

	msg, err := c.readHandshake()
	if err != nil {
		return err
	}
	sessionTicketMsg, ok := msg.(*newSessionTicketMsg)
	if !ok {
		c.sendAlert(alertUnexpectedMessage)
		return unexpectedMessageError(sessionTicketMsg, msg)
	}

	session.sessionTicket = sessionTicketMsg.ticket
	hs.session = session

	hs.writeServerHash(sessionTicketMsg.marshal())

	return nil
}

func (hs *clientHandshakeState) sendFinished(out []byte, isResume bool) error {
	c := hs.c

	var postCCSBytes []byte
	seqno := hs.c.sendHandshakeSeq
	if hs.serverHello.nextProtoNeg {
		nextProto := new(nextProtoMsg)
		proto, fallback := mutualProtocol(c.config.NextProtos, hs.serverHello.nextProtos)
		nextProto.proto = proto
		c.clientProtocol = proto
		c.clientProtocolFallback = fallback

		nextProtoBytes := nextProto.marshal()
		hs.writeHash(nextProtoBytes, seqno)
		seqno++
		postCCSBytes = append(postCCSBytes, nextProtoBytes...)
	}

	if hs.serverHello.channelIDRequested {
		encryptedExtensions := new(encryptedExtensionsMsg)
		if c.config.ChannelID.Curve != elliptic.P256() {
			return fmt.Errorf("tls: Channel ID is not on P-256.")
		}
		var resumeHash []byte
		if isResume {
			resumeHash = hs.session.handshakeHash
		}
		r, s, err := ecdsa.Sign(c.config.rand(), c.config.ChannelID, hs.finishedHash.hashForChannelID(resumeHash))
		if err != nil {
			return err
		}
		channelID := make([]byte, 128)
		writeIntPadded(channelID[0:32], c.config.ChannelID.X)
		writeIntPadded(channelID[32:64], c.config.ChannelID.Y)
		writeIntPadded(channelID[64:96], r)
		writeIntPadded(channelID[96:128], s)
		encryptedExtensions.channelID = channelID

		c.channelID = &c.config.ChannelID.PublicKey

		encryptedExtensionsBytes := encryptedExtensions.marshal()
		hs.writeHash(encryptedExtensionsBytes, seqno)
		seqno++
		postCCSBytes = append(postCCSBytes, encryptedExtensionsBytes...)
	}

	finished := new(finishedMsg)
	if c.config.Bugs.EarlyChangeCipherSpec == 2 {
		finished.verifyData = hs.finishedHash.clientSum(nil)
	} else {
		finished.verifyData = hs.finishedHash.clientSum(hs.masterSecret)
	}
	copy(out, finished.verifyData)
	if c.config.Bugs.BadFinished {
		finished.verifyData[0]++
	}
	c.clientVerify = append(c.clientVerify[:0], finished.verifyData...)
	hs.finishedBytes = finished.marshal()
	hs.writeHash(hs.finishedBytes, seqno)
	postCCSBytes = append(postCCSBytes, hs.finishedBytes...)

	if c.config.Bugs.FragmentAcrossChangeCipherSpec {
		c.writeRecord(recordTypeHandshake, postCCSBytes[:5])
		postCCSBytes = postCCSBytes[5:]
	}
	c.dtlsFlushHandshake()

	if !c.config.Bugs.SkipChangeCipherSpec &&
		c.config.Bugs.EarlyChangeCipherSpec == 0 {
		c.writeRecord(recordTypeChangeCipherSpec, []byte{1})
	}

	if c.config.Bugs.AppDataAfterChangeCipherSpec != nil {
		c.writeRecord(recordTypeApplicationData, c.config.Bugs.AppDataAfterChangeCipherSpec)
	}
	if c.config.Bugs.AlertAfterChangeCipherSpec != 0 {
		c.sendAlert(c.config.Bugs.AlertAfterChangeCipherSpec)
		return errors.New("tls: simulating post-CCS alert")
	}

	if !c.config.Bugs.SkipFinished {
		c.writeRecord(recordTypeHandshake, postCCSBytes)
		c.dtlsFlushHandshake()
	}
	return nil
}

func (hs *clientHandshakeState) writeClientHash(msg []byte) {
	// writeClientHash is called before writeRecord.
	hs.writeHash(msg, hs.c.sendHandshakeSeq)
}

func (hs *clientHandshakeState) writeServerHash(msg []byte) {
	// writeServerHash is called after readHandshake.
	hs.writeHash(msg, hs.c.recvHandshakeSeq-1)
}

func (hs *clientHandshakeState) writeHash(msg []byte, seqno uint16) {
	if hs.c.isDTLS {
		// This is somewhat hacky. DTLS hashes a slightly different format.
		// First, the TLS header.
		hs.finishedHash.Write(msg[:4])
		// Then the sequence number and reassembled fragment offset (always 0).
		hs.finishedHash.Write([]byte{byte(seqno >> 8), byte(seqno), 0, 0, 0})
		// Then the reassembled fragment (always equal to the message length).
		hs.finishedHash.Write(msg[1:4])
		// And then the message body.
		hs.finishedHash.Write(msg[4:])
	} else {
		hs.finishedHash.Write(msg)
	}
}

// clientSessionCacheKey returns a key used to cache sessionTickets that could
// be used to resume previously negotiated TLS sessions with a server.
func clientSessionCacheKey(serverAddr net.Addr, config *Config) string {
	if len(config.ServerName) > 0 {
		return config.ServerName
	}
	return serverAddr.String()
}

// mutualProtocol finds the mutual Next Protocol Negotiation or ALPN protocol
// given list of possible protocols and a list of the preference order. The
// first list must not be empty. It returns the resulting protocol and flag
// indicating if the fallback case was reached.
func mutualProtocol(protos, preferenceProtos []string) (string, bool) {
	for _, s := range preferenceProtos {
		for _, c := range protos {
			if s == c {
				return s, false
			}
		}
	}

	return protos[0], true
}

// writeIntPadded writes x into b, padded up with leading zeros as
// needed.
func writeIntPadded(b []byte, x *big.Int) {
	for i := range b {
		b[i] = 0
	}
	xb := x.Bytes()
	copy(b[len(b)-len(xb):], xb)
}
