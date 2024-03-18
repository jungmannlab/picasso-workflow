#!/usr/bin/env python
"""
Module Name: confluence.py
Author: Heinrich Grabmayr
Initial Date: March 7, 2024
Description: Interaction with Confluence
"""
import logging
import os
import requests

from picasso_workflow.util import AbstractModuleCollection


logger = logging.getLogger(__name__)


class ConfluenceReporter(AbstractModuleCollection):
    """A class to upload reports of automated picasso evaluations
    to confluence
    """

    def __init__(
        self, base_url, space_key, parent_page_title, report_name, token=None
    ):
        logger.debug("Initializing ConfluenceReporter.")
        self.ci = ConfluenceInterface(
            base_url, space_key, parent_page_title, token=token
        )

        # create page
        self.report_page_name = report_name
        # in case the page already exists, add an iteration
        for i in range(1, 30):
            try:
                self.report_page_id = self.ci.create_page(
                    self.report_page_name, body_text=""
                )
                logger.debug(f"Created page {self.report_page_name}")
                break
            except KeyError:
                self.report_page_name = report_name + "_{:02d}".format(i)

    def load_dataset(self, i, pars_load, results_load):
        """Describes the loading
        Args:
            localize_params : dict
                net_gradient : the net gradient used
                frames : the number of frames
        """
        logger.debug("Reporting a loaded dataset.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Load</strong></p>
        <ul>
        <li>Picasso Version: {results_load['picasso version']}</li>
        <li>Movie Location: {pars_load['filename']}</li>
        <li>Analysis Location: {pars_load['save_directory']}</li>
        <li>Movie Size: Frames: {results_load['movie.shape'][0]},
        Width: {results_load['movie.shape'][1]},
        Height: {results_load['movie.shape'][2]}</li>
        <li>Duration: {results_load['duration']} s</li>
        </ul>
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )
        if (sample_mov_res := results_load.get("sample_movie")) is not None:
            text = f"""
            <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
            <p>Subsampled Frames</p>
            <ul>
            <li> {len(sample_mov_res['sample_frame_idx'])} frames:
             {str(sample_mov_res['sample_frame_idx'])}</li>
            </ul>
            </ac:layout-cell></ac:layout-section></ac:layout>
            """
            self.ci.update_page_content(
                self.report_page_name, self.report_page_id, text
            )
            logger.debug("Uploading movie of subsampled images.")
            self.ci.upload_attachment(
                self.report_page_id, sample_mov_res["filename"]
            )
            self.ci.update_page_content_with_movie_attachment(
                self.report_page_name,
                self.report_page_id,
                os.path.split(sample_mov_res["filename"])[1],
            )
        return

    def identify(self, i, pars_identify, results_identify):
        """Describes the identify step
        Args:
            localize_params : dict
                net_gradient : the net gradient used
                frames : the number of frames
            fn_movie : str
                the filename to the movie generated
            fn_hist : str
                the filename to the histogram plot generated
        """
        logger.debug("Reporting Identification.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Identify</strong></p>
        <ul>
        <li>Min Net Gradient: {pars_identify['min_grad']:,.0f}</li>
        <li>Box Size: {pars_identify['box_size']} px</li>
        <li>Duration: {results_identify['duration']} s</li>
        <li>Identifications found: {results_identify['num_identifications']:,}
        </li>
        </ul>
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )
        if (
            res_autonetgrad := results_identify.get("auto_netgrad")
        ) is not None:
            logger.debug("Uploading graph for auto_netgrad.")
            self.ci.upload_attachment(
                self.report_page_id, res_autonetgrad["filename"]
            )
            self.ci.update_page_content_with_image_attachment(
                self.report_page_name,
                self.report_page_id,
                os.path.split(res_autonetgrad["filename"])[1],
            )
        if (res := results_identify.get("ids_vs_frame")) is not None:
            logger.debug("uploading graph for identifications vs frame.")
            self.ci.upload_attachment(self.report_page_id, res["filename"])
            self.ci.update_page_content_with_image_attachment(
                self.report_page_name,
                self.report_page_id,
                os.path.split(res["filename"])[1],
            )

    def localize(self, i, pars_localize, results_localize):
        """Describes the Localize section of picasso
        Args:
            localize_params : dict
                net_gradient : the net gradient used
                frames : the number of frames
        """
        logger.debug("Reporting Localization of spots.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Localize</strong></p>
        <ul><li>Duration: {results_localize['duration']}</li>
        <li>Locs Column names: {results_localize['locs_columns']}</li></ul>
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        # text = "<p><strong>Localize</strong></p>"
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )

        if (res := results_localize.get("locs_vs_frame")) is not None:
            # print('uploading graph')
            self.ci.upload_attachment(self.report_page_id, res["filename"])
            self.ci.update_page_content_with_image_attachment(
                self.report_page_name,
                self.report_page_id,
                os.path.split(res["filename"])[1],
            )

    def undrift_rcc(self, i, pars_undrift, res_undrift):
        """Describes the Localize section of picasso
        Args:
            localize_params : dict
                net_gradient : the net gradient used
                frames : the number of frames
        """
        logger.debug("Reporting undrifting via RCC.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Undrifting via RCC</strong></p>
        <ul><li>Dimensions: {pars_undrift.get('dimensions')}</li>
        <li>Segmentation: {pars_undrift.get('segmentation')}</li>
        """
        if msg := res_undrift.get("message"):
            text += f"""<li>Note: {msg}</li>"""
        text += f"""
        <li>Duration: {res_undrift.get('duration')} s</li></ul>
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )

        if driftimg_fn := pars_undrift.get("drift_image"):
            self.ci.upload_attachment(self.report_page_id, driftimg_fn)
            self.ci.update_page_content_with_image_attachment(
                self.report_page_name,
                self.report_page_id,
                os.path.split(driftimg_fn)[1],
            )

    def manual(self, i, parameters, results):
        """ """
        logger.debug("Reporting manual step")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Manual step</strong></p>
        <ul><li>prompt: {parameters.get('prompt')}</li>
        <ul><li>filename: {parameters.get('filename')}</li>
        <ul><li>file present: {results.get('success')}</li>
        """
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )

    def describe(self, i, pars_describe, res_describe):
        logger.debug("Reporting dataset description.")
        text = """
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Descriptive Statistics</strong></p>"""
        for meth, meth_pars in pars_describe["methods"].items():
            if meth.lower() == "nena":
                meth_res = res_describe["nena"]
                text += f"""
                    <p>NeNa</p>
                    <ul><li>Segmentation: {meth_pars['segmentation']}</li>
                    <li>Best Values: {str(meth_res['best_vals'])}</li>
                    <li>Result: {str(meth_res['res'])}</li>
                    </ul>"""
        text += """
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )


class UndriftError(Exception):
    pass


class ConfluenceInterface:
    """A Interface class to access Confluence

    For access to the Confluence API, create an API token in confluence,
    and store it as an environment variable:
    $ setx CONFLUENCE_BEARER "your_confluence_api_token"
    """

    def __init__(self, base_url, space_key, parent_page_title, token=None):
        if token is None:
            self.bearer_token = self.get_bearer_token()
        else:
            self.bearer_token = token
        self.base_url = base_url
        self.space_key = space_key
        self.parent_page_id, _ = self.get_page_properties(parent_page_title)

    def get_bearer_token(self):
        """Set this by setting the environment variable in the windows command
        line on the server:
        $ setx CONFLUENCE_BEARER <your_confluence_api_token>
        The confluence api token can be generated and copied in the personal
        details of confluence.
        """
        return os.environ.get("CONFLUENCE_BEARER")

    def get_page_properties(self, page_title="", page_id=""):
        """
        Returns:
            id : str
                the page id
            title : str
                the page title
        """
        if page_title != "":
            url = self.base_url + "/rest/api/content"
            params = {"spaceKey": self.space_key, "title": page_title}
        elif page_id != "":
            url = self.base_url + f"/rest/api/content/{page_id}"
            params = {
                "spaceKey": self.space_key,
            }
        else:
            logger.error("One of page_title and page_id must be given.")
        headers = {"Authorization": f"Bearer {self.bearer_token}"}
        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            logger.warn("Failed to get page content.")
        results = response.json()["results"][0]
        return results["id"], results["title"]

    def get_page_version(self, page_title="", page_id=""):
        """
        Returns:
            data : dict
                results
                    id, title, version
        """
        if page_title != "":
            url = self.base_url + "/rest/api/content"
            params = {
                "spaceKey": self.space_key,
                "title": page_title,
            }
        elif page_id != "":
            url = self.base_url + f"/rest/api/content/{page_id}"
            params = {}
        else:
            logger.error("One of page_title and page_id must be given.")
        params["expand"] = ["version"]
        headers = {"Authorization": f"Bearer {self.bearer_token}"}
        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            logger.warn("Failed to get page content.")
        return response.json()["results"][0]["version"]["number"]

    def get_page_body(self, page_title="", page_id=""):
        """
        Returns:
            data : dict
                results
                    id, title, version
        """
        if page_title != "":
            url = self.base_url + "/rest/api/content"
            params = {
                "spaceKey": self.space_key,
                "title": page_title,
            }
        elif page_id != "":
            url = self.base_url + f"/rest/api/content/{page_id}"
            params = {}
        else:
            logger.error("One of page_title and page_id must be given.")
        params["expand"] = ["body.storage"]
        headers = {"Authorization": f"Bearer {self.bearer_token}"}
        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            logger.warn("Failed to get page content.")
        return response.json()["results"][0]["body"]["storage"]["value"]

    def create_page(self, page_title, body_text, parent_id="rootparent"):
        """
        Args:
            page_title : str
                the title of the page to be created
            body_text : str
                the content of the page, with the confuence markdown / html
            parent_id : str
                the id of the parent page. If 'rootparent', the parent_page_id
                of this ConfluenceInterface is used
        Returns:
            page_id : str
                the id of the newly created page
        """
        if parent_id == "rootparent":
            parent_id = self.parent_page_id
        url = self.base_url + "/rest/api/content"
        headers = {
            "Authorization": "Bearer {:s}".format(self.bearer_token),
            "Content-Type": "application/json",
        }
        data = {
            "type": "page",
            "title": page_title,
            "space": {"key": self.space_key},
            "ancestors": [{"id": parent_id}],
            "body": {
                "storage": {"value": body_text, "representation": "storage"}
            },
        }
        response = requests.post(url, headers=headers, json=data)
        if response.status_code != 200:
            logger.warning(f"Failed to create page {page_title}.")
            raise KeyError()

        return response.json()["id"]

    def delete_page(self, page_id):
        url = self.base_url + "/rest/api/content/{page_id}"
        headers = {
            "Authorization": "Bearer {:s}".format(self.bearer_token),
            "Content-Type": "application/json",
        }
        response = requests.delete(url, headers=headers)
        if response.status_code == 204:
            logger.debug("Page deleted successfully.")
        else:
            logger.error("Failed to delete the page.")

    def upload_attachment(self, page_id, filename):
        """Uploads an attachment to a page
        Args:
            page_id : str
                the page id the attachment should be saved to.
            filename : str
                the local filename of the file to attach
        Returns:
            attachment_id : str
                the id of the attachment
        """
        url = self.base_url + f"/rest/api/content/{page_id}/child/attachment"
        headers = {
            "Authorization": "Bearer {:s}".format(self.bearer_token),
            "X-Atlassian-Token": "nocheck",
        }
        with open(filename, "rb") as f:
            files = {"file": f}
            response = requests.post(url, headers=headers, files=files)
        if response.status_code != 200:
            logger.warning("Failed to upload attachment.")
            return

        attachment_id = response.json()["results"][0]["id"]
        return attachment_id

    def update_page_content(self, page_name, page_id, body_update):
        prev_version = self.get_page_version(page_name)
        prev_body = self.get_page_body(page_name)
        _, prev_title = self.get_page_properties(page_name)

        url = self.base_url + f"/rest/api/content/{page_id}"
        headers = {
            "Authorization": "Bearer {:s}".format(self.bearer_token),
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        data = {
            "version": {
                "number": prev_version + 1,
                "message": "version update",
            },
            "type": "page",
            "title": prev_title,
            "body": {
                "storage": {
                    "value": prev_body + body_update,
                    "representation": "storage",
                }
            },
        }
        response = requests.put(url, headers=headers, json=data)
        if response.status_code != 200:
            logger.warning("Failed to update page content.")

    def update_page_content_with_movie_attachment(
        self, page_name, page_id, filename
    ):
        body_update = f"""
            <ac:structured-macro ac:name="multimedia" ac:schema-version="1">
            <ac:parameter ac:name="autoplay">false</ac:parameter>
            <ac:parameter ac:name="name"><ri:attachment
            ri:filename=\"{filename}\" /></ac:parameter>
            <ac:parameter ac:name="loop">false</ac:parameter>
            <ac:parameter ac:name="width">30%</ac:parameter>
            <ac:parameter ac:name="height">30%</ac:parameter>
            </ac:structured-macro>
            """
        self.update_page_content(page_name, page_id, body_update)

    def update_page_content_with_image_attachment(
        self, page_name, page_id, filename
    ):
        body_update = (
            f'<ac:image><ri:attachment ri:filename="{filename}" />'
            + "</ac:image>"
        )
        self.update_page_content(page_name, page_id, body_update)
