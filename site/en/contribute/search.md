Project: /_project.yaml
Book: /_book.yaml

# Searching the codebase

{% include "_buttons.html" %}

## Product overview {:#product-overview}

Bazel's [code search and source browsing interface](https://source.bazel.build)
is a web-based tool for browsing Bazel source code repositories. You can
use these features to navigate among different repositories, branches, and
files. You can also view history, diffs, and blame information.

## Getting started {:#getting-started}

Note: For the best experience, use the latest version of Chrome, Safari, or
Firefox.

To access the code search and source browsing interface, open
[https://source.bazel.build](https://source.bazel.build) in your web browser.

The main screen appears. This screen contains the following components:

1. The Breadcrumb toolbar. This toolbar displays your current location in the
repository and allows you to move quickly to another location such as another
repository, or another location within a repository, such as a file, branch, or
commit.

1. A list of repositories that you can browse.

At the top of the screen is a search box. You can use this box to search for
specific files and code.

## Working with repositories {:#working-with-repositories}

### Opening a repository {:#opening-a-repository}

To open a repository, click its name from the main screen.

Alternatively, you can use the Breadcrumb toolbar to browse for a
specificrepository. This toolbar displays your current location in the
repository and allows you to move quickly to another location such as another
repository, or another location within a repository, such as a file, branch, or
commit.

### Switch repositories {:#switch-repositories}

To switch to a different repository, select the repository from the Breadcrumb toolbar.

### View a repository at a specific commit {:#view-a-repository-at-a-specific-commit}

To view a repository at a specific commit:

1. From the view of the repository, select the file.
1. From the Breadcrumb toolbar, open the **Branch** menu.
1. In the submenu that appears, click **Commit**.
1. Select the commit you want to view.

The interface now shows the repository as it existed at that commit.

### Open a branch, commit, or tag {:#open-a-branch-commit-or-tag}

By default, the code search and source browsing interface opens a repository to
the default branch.  To open a different branch, from the Breadcrumb toolbar,
click the **Branch/Commit/Tag** menu. A submenu opens, allowing you to select a
branch using a branch name, a tag name, or through a search box.

*  To select a branch using a branch name, select **Branch** and then click the
   name of the branch.
*  To select a branch using a tag name, select **Tag** and
   then click the tag name.
*  To select a branch using a commit id, select **Commit** and then click the
   commit id.
*  To search for a branch, commit, or tag, select the corresponding item and
   type a search term in the search box.

## Working with files {:#working-with-files}

When you select a repository from the main screen, the screen changes to display
a view of that repository. If a README file exists, its contents appear in the
file pane, located on the right side of the screen. Otherwise, a list of
repository's files and folders appear.  On the left side of the screen is a tree
view of the repository's files and folders. You can use this tree to browse and
open specific files.

Notice that, when you are viewing a repository, the Breadcrumb toolbar now has
three components:

*  A **Repository** menu, from which you can select different repositories
*  A **Branch/Commit/Tag** menu, from which you can select specific branches,
   tags, or commits
*  A **File path** box, which displays the name of the current file or folder
   and its corresponding path

### Open a file {:#open-a-file}

You can open a file by browsing to its directory and selecting it. The view of
the repository updates to show the contents of the file in the file pane, and
its location in the repository in the tree pane.

### View file changes {:#view-file-changes}

To view file changes:

1. From the view of the repository, select the file.
1. Click **BLAME**, located in the upper-right corner.

The file pane updates to display who made changes to the file and when.

### View change history {:#view-change-history}

To view the change history of a file:

1.  From the view of the repository, select the file.
1.  Click **HISTORY**, located in the upper-right corner.
    The **Change history** pane appears, showing the commits for this file.

### View code reviews {:#view-code-reviews}

For Gerrit code reviews, you can open the tool directly from the Change History pane.

To view the code review for a file:

1. From the view of the repository, select the file.
1. Click **HISTORY**, located in the upper-right corner. The Change History pane
   appears, showing the commits for this file.
1. Hover over a commit. A **More** button (three vertical dots) appears.
1. Click the **More** button.
1. Select **View code review**.

The Gerrit Code Review tool opens in a new browser window.

### Open a file at a specific commit {:#open-a-file-at-a-specific-commit}

To open a file at a specific commit:

1. From the view of the repository, select the file.
1. Click **HISTORY**, located in the upper-right corner. The Change History pane
   appears, showing the commits for this file.
1. Hover over a commit. A **VIEW** button appears.
1. Click the **VIEW** button.

### Compare a file to a different commit {:#compare-a-file-to-a-different-commit}

To compare a file at a different commit:

1. From the view of the repository, select the file. To compare from two
   different commits, first open the file at that commit.
1. Hover over a commit. A **DIFF** button appears.
1. Click the **DIFF** button.

The file pane updates to display a side-by-side comparison between the two
files. The oldest of the two commits is always on the left.

In the Change History pane, both commits are highlighted, and a label indicates
if the commit is displayed on the left or the right.

To change either file, hover over the commit in the Change History pane. Then,
click either the **Left** or **Right** button to have the open the commit on the
left or right side of the diff.

### Browsing cross references {:#browsing-cross-references}

Another way to browse source repositories is through the use of cross
references. These references appear automatically as hyperlinks within a given
source file.

To make cross references easier to identify, click **Cross References**,
located in the upper-right corner. This option displays an underline below all
cross references in a file.

**Note:** If **Cross References** is grayed out, it indicates that
cross references are not available for that file.

Click a cross reference to open the Cross Reference pane. This pane contains
two sections:

* A **Definition** section, which lists the file or files that define the
  reference
* A **References** section, which lists the files in which the reference also
  appears

Both sections display the name of the file, as well as the line or lines
that contains the reference. To open a file from the Cross Reference pane,
click the line number entry. The file appears in a new section of the pane,
allowing you to continue to browse the file while keeping the original file
in view.

You can continue to browse cross references using the Cross Reference pane, just
as you can in the File pane. When you do, the pane displays a breadcrumb trail,
which you can use to navigate between different cross references.

## Searching for code {:#search}

You can search for specific files or code snippets using the search box located
at the top of the screen. Searches are always against the default branch.

All searches use [RE2 regular expressions](https://github.com/google/re2/wiki/Syntax){: .external}
by default. If you do not want to use regular expressions, enclose your search
in double quotes ( " ).

**Note:** To quickly search for a specific file, either add a backslash in front
of the period, or enclose the entire file name in quotes.

```
foo\.java
"foo.java"
```

You can refine your search using the following filters.

<table border="1px">
<thead>
<tr>
<th style="padding:5px"><strong>Filter</strong></th>
<th style="padding:5px"><strong>Other options</strong></th>
<th style="padding:5px"><strong>Description</strong></th>
<th style="padding:5px"><strong>Example</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td style="padding:5px">lang:</td>
<td style="padding:5px">language:</td>
<td style="padding:5px">Perform an exact match by file language.</td>
<td style="padding:5px">lang:java test</td>
</tr>
<tr>
<td style="padding:5px">file:</td>
<td style="padding:5px">filepath:<br>
path:<br>
f:</td>
<td style="padding:5px"></td>
<td style="padding:5px"></td>
</tr>
<tr>
<td style="padding:5px">case:yes</td>
<td style="padding:5px"></td>
<td style="padding:5px">Make the search case sensitive. By default, searches are not case-sensitive.</td>
<td style="padding:5px">case:yes Hello World</td>
</tr>
<tr>
<td style="padding:5px">class:</td>
<td style="padding:5px"></td>
<td style="padding:5px">Search for a class name.</td>
<td style="padding:5px">class:MainClass</td>
</tr>
<tr>
<td style="padding:5px">function:</td>
<td style="padding:5px">func:</td>
<td style="padding:5px">Search for a function name.</td>
<td style="padding:5px">function:print</td>
</tr>
<tr>
<td style="padding:5px">-</td>
<td style="padding:5px"></td>
<td style="padding:5px">Negates the term from the search.</td>
<td style="padding:5px">hello -world</td>
</tr>
<tr>
<td style="padding:5px">\</td>
<td style="padding:5px"></td>
<td style="padding:5px">Escapes special characters, such as ., \, or (.</td>
<td style="padding:5px">run\(\)</td>
</tr>
<tr>
<td style="padding:5px">"[term]"</td>
<td style="padding:5px"></td>
<td style="padding:5px">Perform a literal search.</td>
<td style="padding:5px">"class:main"</td>
</tr>
</tbody>
</table>

## Additional Support {:#additional-support}

To report an issue, click the **Feedback** button that appears in the top
right-hand corner of the screen and enter your feedback in the provided form.
