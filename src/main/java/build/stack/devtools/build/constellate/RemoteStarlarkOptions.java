package build.stack.devtools.build.constellate;

import com.google.common.flogger.GoogleLogger;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.List;

/** Options for remote starlark. */
public class RemoteStarlarkOptions extends OptionsBase {
    private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

    @Option(name = "listen_port", defaultValue = "3524", category = "build_worker", documentationCategory = OptionDocumentationCategory.UNCATEGORIZED, effectTags = {
            OptionEffectTag.UNKNOWN }, help = "Listening port for the netty server.")
    public int listenPort;

    @Option(name = "debug", defaultValue = "false", category = "build_worker", documentationCategory = OptionDocumentationCategory.UNCATEGORIZED, effectTags = {
            OptionEffectTag.UNKNOWN }, help = "Turn this on for debugging remote job failures. There will be extra messages and the "
                    + "work directory will be preserved in the case of failure.")
    public boolean debug;

    @Option(name = "log_level", defaultValue = "INFO", category = "build_worker", documentationCategory = OptionDocumentationCategory.UNCATEGORIZED, effectTags = {
            OptionEffectTag.UNKNOWN }, help = "Set the logging level (SEVERE, WARNING, INFO, CONFIG, FINE, FINER, FINEST, ALL)")
    public String logLevel;

    @Option(name = "pid_file", defaultValue = "null", category = "build_worker", documentationCategory = OptionDocumentationCategory.UNCATEGORIZED, effectTags = {
            OptionEffectTag.UNKNOWN }, help = "File for writing the process id for this worker when it is fully started.")
    public String pidFile;

    @Option(name = "http_listen_port", defaultValue = "0", category = "build_worker", documentationCategory = OptionDocumentationCategory.UNCATEGORIZED, effectTags = {
            OptionEffectTag.UNKNOWN }, help = "Starts an embedded HTTP REST server on the given port. The server will simply store PUT "
                    + "requests in memory and return them again on GET requests. This is useful for " + "testing only.")
    public int httpListenPort;

    @Option(name = "tls_certificate", defaultValue = "null", documentationCategory = OptionDocumentationCategory.UNCATEGORIZED, effectTags = {
            OptionEffectTag.UNKNOWN }, help = "Specify the TLS server certificate to use.")
    public String tlsCertificate;

    @Option(name = "tls_private_key", defaultValue = "null", documentationCategory = OptionDocumentationCategory.UNCATEGORIZED, effectTags = {
            OptionEffectTag.UNKNOWN }, help = "Specify the TLS private key to be used.")
    public String tlsPrivateKey;

    @Option(name = "tls_ca_certificate", defaultValue = "null", documentationCategory = OptionDocumentationCategory.UNCATEGORIZED, effectTags = {
            OptionEffectTag.UNKNOWN }, help = "Specify a CA certificate to use for authenticating clients; setting this implicitly "
                    + "requires client authentication (aka mTLS).")
    public String tlsCaCertificate;

    private static final int MAX_JOBS = 16384;

}
