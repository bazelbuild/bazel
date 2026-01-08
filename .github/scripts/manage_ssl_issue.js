module.exports = async ({github, context}) => {
  const fs = require("fs");
  const output = fs.readFileSync("ssl_output.txt", "utf8");
  const title = "🚨 Urgent: SSL Certificates Expiring";

  // Find existing open issues created by this workflow
  const issues = await github.rest.issues.listForRepo({
    owner: context.repo.owner,
    repo: context.repo.repo,
    state: "open",
    creator: "github-actions[bot]"
  });

  const existingIssue = issues.data.find(i => i.title === title);
  const body = [
    "### SSL Certificate Warning",
    "",
    "The automated monitor has detected SSL issues.",
    "",
    "#### Details:",
    "```",
    output,
    "```",
    "",
    "**Action Required:**",
    "1. Check `.github/config/ssl_domains.yaml` to verify the domain list.",
    "2. Renew the certificates.",
    "",
    `[Workflow Run Log](${context.serverUrl}/${context.repo.owner}/${context.repo.repo}/actions/runs/${context.runId})`
  ].join("\n");

  if (existingIssue) {
    await github.rest.issues.createComment({
      owner: context.repo.owner,
      repo: context.repo.repo,
      issue_number: existingIssue.number,
      body: "SSL check failed again. Latest status:\n\n" + body
    });
  } else {
    await github.rest.issues.create({
      owner: context.repo.owner,
      repo: context.repo.repo,
      title: title,
      body: body,
      labels: ["breakage", "P0", "team-OSS"]
    });
  }
}
